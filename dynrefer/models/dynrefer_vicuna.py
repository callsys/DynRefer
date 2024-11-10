import math
import copy
import os.path
import random
import einops
import itertools
import matplotlib.pyplot as plt
import json
import pickle
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.distributed as dist
from types import MethodType
from torch.nn import functional as F
from textblob import TextBlob
from peft import LoraConfig, get_peft_model

from lavis.models.base_model import concat_all_gather
from lavis.common.registry import registry
from dynrefer.models.blip2_models.blip2_vicuna_instruct import Blip2VicunaInstruct
from dynrefer.models.tagging_heads.bert import BertConfig, BertModel
from dynrefer.models.tagging_heads.asymmetric_loss import AsymmetricLoss
from torchvision.models.vision_transformer import MLPBlock
from functools import partial


from DCNv4.functions import DCNv4Function
from torch.nn.init import xavier_uniform_, constant_


class CustomDCNv4(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            dw_kernel_size=3,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs):
        """
        DCNv4 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group

        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_group % 16 == 0

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.K =  group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(channels*2, channels, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels)
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3)/8)*8))
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.)
        constant_(self.offset_mask.bias.data, 0.)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, ref, input):        
        N, C, H, W = input.shape
        input = input.reshape(N, C, -1).permute(0, 2, 1)
        x = input
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1).contiguous()
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(torch.cat([input.view(N, H, W, C).permute(0, 3, 1, 2), ref], dim=1))
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, -1, C)
        else:
            offset_mask_input = input
        offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)

        x_proj = x

        x = DCNv4Function.apply(
            x, offset_mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center
            )

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.view(N, -1, C)

        if not self.without_pointwise:
            x = self.output_proj(x)
        x = x.reshape(N, H, W, C).permute(0, 3, 1, 2)
        
        return x



@registry.register_model("dynrefer_vicuna")
class DynReferVicuna(Blip2VicunaInstruct):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        base_kwargs = copy.deepcopy(kwargs)
        base_kwargs_keys = ["vit_model", "img_size", "drop_path_rate", "use_grad_checkpoint", "vit_precision",
                            "freeze_vit", "num_query_token", "llm_model", "prompt", "max_txt_len", "max_output_txt_len"
                            "qformer_text_input", "apply_lemmatizer"]
        for key in kwargs.keys():
            if key not in base_kwargs_keys:
                base_kwargs.pop(key)
        super().__init__(*args, **base_kwargs)

        num_views = kwargs.get("num_views", 2)
        self.num_views = num_views
        vit_model = kwargs.get("vit_model", "eva_clip_g")
        self.vit_model = vit_model

        # contextual visual embedding module
        if vit_model == "eva_clip_g":
            input_image_size = self.visual_encoder.image_size
            patch_size = self.visual_encoder.patch_embed.patch_size[0]
            vision_embed_dim = self.visual_encoder.embed_dim
        else:
            # vit-large model
            input_image_size = self.visual_encoder.input_resolution
            patch_size = self.visual_encoder.conv1.kernel_size[0]
            vision_embed_dim = self.visual_encoder.num_features
        self._roi_align = torchvision.ops.RoIAlign(output_size=input_image_size // patch_size,
                                                   spatial_scale=1 / patch_size,
                                                   sampling_ratio=2)
        
        self.cvem_align =  CustomDCNv4(vision_embed_dim)
        self.cvem_mlp = nn.Sequential(
            nn.Linear(vision_embed_dim * num_views, vision_embed_dim),
            nn.ReLU(),
            nn.Linear(vision_embed_dim, vision_embed_dim))
        # control embedding module
        self.cem_memory = nn.Parameter(torch.zeros(self.llm_model.lm_head.in_features))

        # clip text encoder
        self.build_clip_text_encoder()

        # region tagging
        self.build_tag_embeds()
        embed_dim = 256
        tag_bert_config = BertConfig.from_json_file(
            kwargs.get("tag_bert_config", "dynrefer/models/model_configs/tag_bert_config.json"))
        tag_bert_config.encoder_width = embed_dim
        tag_bert_config.hidden_size = embed_dim
        tag_bert_config.intermediate_size = embed_dim * 4
        self.tag_head = BertModel(config=tag_bert_config, add_pooling_layer=False)
        del self.tag_head.embeddings
        for layer in self.tag_head.encoder.layer:
            del layer.attention
        self.tag_vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.tag_text_proj = nn.Linear(self.clip_embeds.shape[-1], embed_dim)
        self.tag_fc = nn.Linear(tag_bert_config.hidden_size, 1)
        self.tag_weight = 0.01
        self.tag_loss_function = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)

        # attribute
        self.build_att_embeds()
        self.attr_weight = 0.5
        self.attr_head = BertModel(config=tag_bert_config, add_pooling_layer=False)
        del self.attr_head.embeddings
        for layer in self.attr_head.encoder.layer:
            del layer.attention
        self.attr_text_proj = nn.Linear(self.att_embeds.shape[-1], embed_dim)
        self.attr_vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.attr_fc = nn.Linear(tag_bert_config.hidden_size, 1)
        self.attr_t = nn.Parameter(math.log(10) * torch.ones([]))
        self.attr_b = nn.Parameter(-10 * torch.ones([]))

        # open-vocabulary
        self.build_ov_embeds()

        # Trainable parameters
        names = ["cvem", "cem", "tag", "attr", "Qformer", "llm_proj"]
        self.finetune_llm = kwargs.get("finetune_llm", False)
        if self.finetune_llm:
            lora_config = LoraConfig(
                r=64, lora_alpha=128, lora_dropout=0.0,
                target_modules=["embed_tokens", "lm_head", "q", "v"]
            )

            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.to(torch.float32)
            names.extend(["lora"])
        params = [0] * len(names)

        trainable_params = 0
        all_params = 0
        for param_name, param in self.named_parameters():
            all_params += param.numel()
            param.requires_grad = False
            for idx, name in enumerate(names):
                if name in param_name:
                    param.requires_grad = True
                    trainable_params += param.numel()
                    params[idx] += param.numel()
                    break
        print(f"[trainable ratio : {trainable_params / all_params}]")
        for idx, name in enumerate(names):
            print(f"[{name} ratio : {params[idx] / all_params}]")

    def build_clip_text_encoder(self):
        if self.vit_model == "eva_clip_g":
            import torch.nn as nn
            from dynrefer.models.eva_clip.clip import tokenize
            from dynrefer.models.eva_clip.eva_model import TextTransformer
            text_encoder = TextTransformer(
                vocab_size=49408,
                width=768,
                layers=12,
                heads=12,
                context_length=77,
                embed_dim=1024,
                act_layer=nn.GELU)
            state_dict = torch.load("ckpts/eva_clip_psz14.pt", map_location="cpu")
            state_dict = {k[5:]: v for k, v in state_dict.items() if "text" in k}
            incompatible_keys = text_encoder.load_state_dict(state_dict, strict=False)
            self.text_encoder = text_encoder.to("cuda")
            self.text_tokenize = tokenize
        else:
            from transformers import CLIPTextModel, CLIPTokenizer
            self.text_encoder = CLIPTextModel.from_pretrained("ckpts/clip-vit-large-patch14", torch_dtype=torch.float32).to("cuda")
            self.text_tokenize = CLIPTokenizer.from_pretrained("ckpts/clip-vit-large-patch14", torch_dtype=torch.float32)

    def build_tag_embeds(self):
        tag_list = self.kwargs.get("tag_list", "dynrefer/models/model_configs/ram_tag_list.txt")
        with open(tag_list, "r") as fr:
            self.tag_list = fr.readlines()
        self.tag_list = [tag.strip() for tag in self.tag_list]
        self.num_tags = len(self.tag_list)
        clip_embeds = []
        bs = 64
        bs_tag_list = [self.tag_list[i:i+bs] for i in range(0, len(self.tag_list), bs)]
        for tags in tqdm.tqdm(bs_tag_list):
            clip_embed = self.encode_text(tags)
            clip_embeds.append(clip_embed.detach().cpu())
        clip_embeds = torch.cat(clip_embeds, 0)
        self.clip_embeds = torch.nn.Parameter(clip_embeds)

    def build_att_embeds(self):
        ovad_file = self.kwargs.get("ovad_file", "dynrefer/models/model_configs/ovad2000.json")
        ovad = json.load(open(ovad_file, "r"))
        object_word = "object" if False else ""
        use_prompts = ["the"]
        from dynrefer.common.evaluation.ovad.misc import object_attribute_templates, ovad_validate

        # unconditional embeddings
        all_att_templates = []
        all_att_syns = []
        for att_dict in ovad["attributes"]:
            att_w_type = att_dict["name"]
            att_type, att_list = att_w_type.split(":")
            is_has = att_dict["is_has_att"]
            dobj_name = (
                att_type.replace(" tone", "")
                # So far only for tone worked to remove the word
                # .replace(" color", "")
                # .replace(" pattern", "")
                # .replace(" expression", "")
                # .replace(" type", "")
                # .replace(" length", "")
            )

            # extend the maturity to include other words
            if att_list == "young/baby":
                att_list += "/kid/kids/child/toddler/boy/girl"
            elif att_list == "adult/old/aged":
                att_list += "/teen/elder"
            att_templates = []
            for syn in att_list.split("/"):
                for prompt in use_prompts:
                    for template in object_attribute_templates[is_has][prompt]:
                        if is_has == "has":
                            att_templates.append(
                                template.format(
                                    attr=syn, dobj=dobj_name, noun=object_word
                                ).strip()
                            )
                        elif is_has == "is":
                            att_templates.append(
                                template.format(attr=syn, noun=object_word).strip()
                            )
                        all_att_syns.append(syn)

            all_att_templates.append(att_templates)

        len_synonyms = [len(att_synonyms) for att_synonyms in all_att_templates]
        all_att_templates = list(itertools.chain.from_iterable(all_att_templates))

        att_embeds = []
        bs = 64
        att_list = [all_att_templates[i:i + bs] for i in range(0, len(all_att_templates), bs)]
        for att in tqdm.tqdm(att_list):
            att_embed = self.encode_text(att)
            att_embeds.append(att_embed.detach().cpu())

        att_embeds = torch.cat(att_embeds, 0)

        self.att_embeds = torch.nn.Parameter(att_embeds)
        self.syn2template = dict()
        for syn, template in zip(all_att_syns, all_att_templates):
            if syn not in self.syn2template:
                self.syn2template[syn] = [template]
            else:
                self.syn2template[syn].append(template)
        self.att_len_synonyms = len_synonyms

    def build_ov_embeds(self):
        ov_file = self.kwargs.get("ov_file", "dynrefer/models/model_configs/coco.json")
        ov_classes = json.load(open(ov_file, "r"))
        prompt = "the {}"
        self.ov_classes = ov_classes
        ov_classes = [prompt.format(class_text.lower()) for class_text in ov_classes]
        ov_embeds = []
        bs = 64
        ov_list = [ov_classes[i:i + bs] for i in range(0, len(ov_classes), bs)]
        for vocab in tqdm.tqdm(ov_list):
            ov_embed = self.encode_text(vocab)
            ov_embeds.append(ov_embed.detach().cpu())
        ov_embeds = torch.cat(ov_embeds, 0)
        self.ov_embeds = torch.nn.Parameter(ov_embeds)

    def encode_text(self, texts):
        if self.vit_model == "eva_clip_g":
            clip_embed = self.text_tokenize(texts).to("cuda")
            clip_embed = self.text_encoder(clip_embed)
        else:
            text_input = self.text_tokenize(texts,
                                            padding="max_length",
                                            max_length=self.text_tokenize.model_max_length,
                                            truncation=True,
                                            return_tensors="pt")
            clip_embed = self.text_encoder(text_input.input_ids.to("cuda")).pooler_output
        return clip_embed

    def roi_align(self, embeds, region_bboxes):
        # prepare cls image embeds and spatio image embeddings
        spatio_image_embeds = embeds[:, 1:]
        cls_image_embeds = embeds[:, 0][:, None]
        b, hw, c = spatio_image_embeds.shape
        ns = int(b/self.num_views)
        h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
        spatio_image_embeds = spatio_image_embeds.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # extract roi features
        bboxes = region_bboxes
        ids = torch.arange(len(bboxes)).to(bboxes).to(torch.int64)
        rois = torch.cat([ids[:, None], bboxes], -1)
        spatio_rois_embeds = self._roi_align(spatio_image_embeds, rois)

        # alignment
        spatio_rois_embeds = einops.rearrange(spatio_rois_embeds, "(s v) c h w -> v s c h w", v=self.num_views, s=ns)
        target = spatio_rois_embeds[1:].reshape(-1, c, h, w)
        source = spatio_rois_embeds[0:1].repeat(self.num_views -1, 1, 1, 1, 1).reshape(-1, c, h, w)
        target = self.cvem_align(source, target)
        spatio_rois_embeds = torch.cat([spatio_rois_embeds[0], target], dim=0)
        spatio_rois_embeds = einops.rearrange(spatio_rois_embeds, "(v s) c h w -> (s v) c h w", v=self.num_views, s=ns)

        cls_image_embeds = cls_image_embeds[ids]

        # back to sequence
        bv = spatio_rois_embeds.shape[0]
        spatio_rois_embeds = spatio_rois_embeds.permute(0, 2, 3, 1).reshape(bv, -1, c)
        rois_embeds = torch.cat([cls_image_embeds, spatio_rois_embeds], 1)
        return rois_embeds

    def cvem_forward(self, samples, embeds):
        ns, nv, c, h, w = samples["cascade_region_images"].shape
        region_bboxes = samples["cascade_region_bboxes"].reshape(-1, 4)

        visual_embeds = self.roi_align(embeds, region_bboxes)
        visual_embeds = einops.rearrange(visual_embeds, "(s v) l c -> s l (v c)", v=nv, s=ns)
        visual_embeds = self.cvem_mlp(visual_embeds)

        return visual_embeds
    
    def tag_forward(self, samples, embeds):
        tag_thr = self.kwargs.get("tag_thr", 0.7)
        bs = len(embeds)
        object_atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        tag_vision_embeds = self.tag_vision_proj(embeds)
        tag_text_embeds = self.tag_text_proj(self.clip_embeds)
        tag_text_embeds = tag_text_embeds.unsqueeze(0).repeat(bs, 1, 1)

        tag_embeds = self.tag_head(
            encoder_embeds=tag_text_embeds,
            encoder_hidden_states=tag_vision_embeds,
            encoder_attention_mask=object_atts,
            return_dict=False,
            mode='tagging',
        )
        tag_logits = self.tag_fc(tag_embeds[0]).squeeze(-1)

        if self.training:
            tags = samples["tags"].to(torch.long)
            tags = (tags[:, :self.num_tags] + tags[:, self.num_tags:]).clip(max=1,min=0)
            loss_tag = self.tag_loss_function(tag_logits, tags) * self.tag_weight
            return loss_tag
        else:
            tag_scores = tag_logits.sigmoid()
            tag_idxs = (tag_scores > tag_thr).to(torch.long)
            tags = [[self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                     for bz_idx in range(len(tag_idxs))]
            samples["pred_tags"] = tags
            return tags

    def att_forward(self, samples, embeds):
        bs = len(embeds)

        att_vision_embeds = self.attr_vision_proj(embeds)

        if self.training:
            att_text_embeds = self.attr_text_proj(self.encode_text(samples["caps"]))

            att_vision_embeds_all = concat_all_gather(
                att_vision_embeds)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
            att_text_embeds_all = concat_all_gather(att_text_embeds)  # [batch_size*num_gpu, embed_dim]

            sim_i2t_embeds = self.attr_head(
                encoder_embeds=att_text_embeds.unsqueeze(0).repeat(len(att_vision_embeds_all), 1, 1),
                encoder_hidden_states=att_vision_embeds_all,
                encoder_attention_mask=torch.ones(att_vision_embeds_all.size()[:-1], dtype=torch.long).to(embeds.device),
                return_dict=False,
                mode='tagging',
            )
            sim_i2t = self.attr_fc(F.normalize(sim_i2t_embeds[0], dim=-1)).squeeze(-1).permute(1, 0)

            sim_t2i_embeds = self.attr_head(
                encoder_embeds=att_text_embeds_all.unsqueeze(0).repeat(len(att_vision_embeds), 1, 1),
                encoder_hidden_states=att_vision_embeds,
                encoder_attention_mask=torch.ones(att_vision_embeds.size()[:-1], dtype=torch.long).to(embeds.device),
                return_dict=False,
                mode='tagging',
            )
            sim_t2i = self.attr_fc(F.normalize(sim_t2i_embeds[0], dim=-1)).squeeze(-1)

            try:
                labels = 2 * torch.eye(bs, device=sim_i2t.device) - 1
                rank = dist.get_rank()
                pad_labels = -torch.ones_like(sim_i2t).to(sim_i2t)
                pad_labels[:, (rank * bs):(rank * bs + bs)] = labels
            except:
                pad_labels = 2 * torch.eye(bs, device=sim_i2t.device) - 1

            logits_i2t = sim_i2t * self.attr_t.exp() + self.attr_b
            loss_att_i2t = -torch.sum(F.logsigmoid(pad_labels * logits_i2t)) / bs

            logits_t2i = sim_t2i * self.attr_t.exp() + self.attr_b
            loss_att_t2i = -torch.sum(F.logsigmoid(pad_labels * logits_t2i)) / bs

            loss_att = (loss_att_i2t + loss_att_t2i) / 2

            return loss_att * self.attr_weight
        else:
            try:
                att_text_embeds = self.attr_text_proj(self.att_embeds.to(self.device))
                sim_i2t_embeds = self.attr_head(
                    encoder_embeds=att_text_embeds.unsqueeze(0).repeat(len(att_vision_embeds), 1, 1),
                    encoder_hidden_states=att_vision_embeds,
                    encoder_attention_mask=torch.ones(att_vision_embeds.size()[:-1], dtype=torch.long).to(embeds.device),
                    return_dict=False,
                    mode='tagging',
                )
                sim_i2t = self.attr_fc(F.normalize(sim_i2t_embeds[0], dim=-1)).squeeze(-1).sigmoid()

                x_attrs_syn = sim_i2t.split(self.att_len_synonyms, dim=1)
                x_attrs_maxsyn = []
                x_attrs_idxsyn = []
                for x_syn in x_attrs_syn:
                    xmax_val, xmax_idx = x_syn.max(axis=1)
                    x_attrs_maxsyn.append(xmax_val)
                    x_attrs_idxsyn.append(xmax_idx)
                attrs = torch.stack(x_attrs_maxsyn, axis=1)
                attrs = attrs.tolist()
            except:
                attrs = [[0]*len(self.att_len_synonyms)]*bs
            return attrs

    def ov_forward(self, samples, embeds):
        att_vision_embeds = self.attr_vision_proj(embeds)
        att_text_embeds = self.attr_text_proj(self.ov_embeds.to(self.device))
        sim_i2t_embeds = self.attr_head(
            encoder_embeds=att_text_embeds.unsqueeze(0).repeat(len(att_vision_embeds), 1, 1),
            encoder_hidden_states=att_vision_embeds,
            encoder_attention_mask=torch.ones(att_vision_embeds.size()[:-1], dtype=torch.long).to(embeds.device),
            return_dict=False,
            mode='tagging',
        )
        sim_i2t = self.attr_fc(F.normalize(sim_i2t_embeds[0], dim=-1)).squeeze(-1)

        clss = sim_i2t.detach().cpu().tolist()
        return clss

    def llm_forward(self, samples, embeds):
        control_words = samples.get("cap_control_words", [""])
        if self.training:
            device = embeds.device
            inputs_llm = self.llm_proj(embeds)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

            self.llm_tokenizer.padding_side = "right"
            self.llm_tokenizer.truncation_side = 'left'
            text_input_tokens = self.llm_tokenizer(
                control_words,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(device)

            self.llm_tokenizer.truncation_side = 'right'
            text_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in samples['caps']],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_output_txt_len,
            ).to(device)

            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens.input_ids,
                text_input_tokens.attention_mask,
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )

            # do not apply loss to the padding
            targets = llm_tokens['input_ids'].masked_fill(
                llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
            )

            # do not apply loss to the text input (i.e., instruction)
            for i, l in enumerate(input_part_targets_len):
                targets[i][:l] = -100

            # do not apply loss to the query tokens
            empty_targets = (
                torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
            )
            targets = torch.cat([empty_targets, targets], dim=1)

            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

            with self.maybe_autocast():
                outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )

            loss_llm = outputs.loss

            return loss_llm
        else:
            device = embeds.device
            self.llm_tokenizer.padding_side = "left"
            inputs_llm = self.llm_proj(embeds)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)

            llm_tokens = self.llm_tokenizer(
                control_words,
                padding="longest",
                return_tensors="pt"
            ).to(device)

            with self.maybe_autocast():
                inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                llm_kwargs = copy.deepcopy(self.kwargs)
                llm_kwargs_keys = ["do_sample", "top_p", "temperature", "num_beams", "max_length",
                                   "min_length", "repetition_penalty", "length_penalty", "num_return_sequences"]
                for key in self.kwargs.keys():
                    if key not in llm_kwargs_keys:
                        llm_kwargs.pop(key)

                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **llm_kwargs
                )

            sequences = outputs.sequences
            scores = outputs.sequences_scores
            scores = torch.exp(scores).cpu().numpy().tolist()

            sequences[sequences == 0] = 2  # convert output id 0 to 2 (eos_token_id)
            captions = self.llm_tokenizer.batch_decode(sequences, skip_special_tokens=True)
            captions = [caption.strip() for caption in captions]

            return {"scores": scores, "caps": captions}

    def parse_sentence(self, samples, tags=None):
        if self.training:
            # attribute extraction
            pos_templates_all = []
            for bz_idx, cap in enumerate(samples["caps"]):
                pos_syns = [syn for syn in self.syn2template if syn in cap]
                pos_templates = []
                for pos_syn in pos_syns:
                    pos_templates.extend(self.syn2template[pos_syn])
                pos_templates_all.append(pos_templates)
            samples["attr_pos_templates"] = pos_templates_all

            # control word extraction
            control_words = []
            full_drop_ratio = self.kwargs.get("full_drop_ratio", 0.5)
            drop_ratio = self.kwargs.get("drop_ratio", 0.5)
            for bz_idx, cap in enumerate(samples["caps"]):
                try:
                    s2 = TextBlob(cap).tags
                    tokens = [el[0] for el in s2]
                    infowords = [name for name, value in s2 if ("NN" in value) or ("JJ" in value)]
                    nouns = [name for name, value in s2 if ("NN" in value)]
                    if len(infowords) > 0:
                        words = []
                        for word in infowords:
                            st_idx = tokens.index(word)
                            ed_idx = st_idx + 1
                            while (ed_idx < len(tokens)) and (tokens[ed_idx] in nouns):
                                ed_idx = ed_idx + 1
                            word = " ".join(tokens[st_idx:ed_idx])
                            words.append(word)
                    else:
                        words = [""]
                except:
                    words = [""]
                tag_idxs = samples["tags"][:, :2*self.num_tags]
                stags = [self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][:self.num_tags])]
                otags = [self.tag_list[tag_idx] for tag_idx in torch.nonzero(tag_idxs[bz_idx][self.num_tags:])]
                tags = stags + otags + words
                tags = list(set(tags))
                l = len(tags)
                if np.random.uniform(0, 1) < full_drop_ratio:
                    control_word = ""
                else:
                    if l == 0:
                        control_word = ""
                    else:
                        sl = torch.from_numpy(np.random.uniform(0, 1, l) > drop_ratio)
                        control_word = [tags[tag_idx] for tag_idx in torch.nonzero(sl)]
                        random.shuffle(control_word)
                        control_word = ",".join(control_word)
                control_words.append(control_word + "|")
            samples["cap_control_words"] = control_words
        else:
            # control word extraction
            control_words = []
            tags = samples.get("pred_tags", tags)
            first_word_control = self.kwargs.get("first_word_control", False)
            if first_word_control:
                first_words = []
                for bz_idx, cap in enumerate(samples["caps"]):
                    try:
                        s2 = TextBlob(cap).tags
                        tokens = [el[0] for el in s2]
                        infowords = [name for name, value in s2 if ("NN" in value) or ("JJ" in value)]
                        nouns = [name for name, value in s2 if ("NN" in value)]
                        if len(infowords) > 0:
                            words = []
                            for word in infowords:
                                st_idx = tokens.index(word)
                                ed_idx = st_idx + 1
                                while (ed_idx < len(tokens)) and (tokens[ed_idx] in nouns):
                                    ed_idx = ed_idx + 1
                                word = " ".join(tokens[st_idx:ed_idx])
                                words.append(word)
                        else:
                            words = []
                    except:
                        words = []
                    if len(words) > 0:
                        first_word = [words[0]]
                    else:
                        first_word = []
                    first_words.append(first_word)
                tags = [fword + tag for fword, tag in zip(first_words, tags)]
            controls = samples.get("controls", None)
            if controls is not None:
                tags = [control + tag for control, tag in zip(controls, tags)]
            for control_tag in tags:
                control_tag = list(set(control_tag))
                control_word = ",".join(control_tag)
                control_words.append(control_word + "|")
            samples["cap_control_words"] = control_words

    def forward(self, samples):
        cascade_region_images = samples["cascade_region_images"]
        ns, nv, c, h, w = cascade_region_images.shape
        image = cascade_region_images.reshape(-1, c, h, w)

        with self.maybe_autocast(dtype=torch.float16):
            embeds = self.ln_vision(self.visual_encoder(image))

        visual_embeds = self.cvem_forward(samples, embeds)
        object_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.query_tokens.expand(visual_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=visual_embeds,
            encoder_attention_mask=object_atts,
            return_dict=True,
        )
        self.parse_sentence(samples)

        with self.maybe_autocast(dtype=torch.bfloat16):
            loss_tag = self.tag_forward(samples, query_output.last_hidden_state)
            loss_att = self.att_forward(samples, query_output.last_hidden_state)
            loss_llm = self.llm_forward(samples, query_output.last_hidden_state)

            return {"loss": loss_llm + loss_tag + loss_att,
                    "loss_llm": loss_llm.detach(),
                    "loss_tag": loss_tag.detach(),
                    "loss_att": loss_att.detach()}
        
    def predict_answers(
            self,
            samples,
            *args,
            **kwargs,
    ):
        split_size = int(self.kwargs.get("split_size", 20))
        l_samples = len(samples["ids"])
        if l_samples > split_size:
            return self.predict_answers_memory_efficient(samples, *args, **kwargs)

        cascade_region_images = samples["cascade_region_images"]
        ns, nv, c, h, w = cascade_region_images.shape
        image = cascade_region_images.reshape(-1, c, h, w)

        with self.maybe_autocast(dtype=torch.float16):
            embeds = self.ln_vision(self.visual_encoder(image))

        visual_embeds = self.cvem_forward(samples, embeds.to(torch.float32))
        object_atts = torch.ones(visual_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.query_tokens.expand(visual_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=visual_embeds,
            encoder_attention_mask=object_atts,
            return_dict=True,
        )

        with self.maybe_autocast(dtype=torch.bfloat16):
            tags = self.tag_forward(samples, query_output.last_hidden_state)
            atts = self.att_forward(samples, query_output.last_hidden_state)
            clss = self.ov_forward(samples, query_output.last_hidden_state)
            self.parse_sentence(samples, tags)
            caps = self.llm_forward(samples, query_output.last_hidden_state)

        output = []
        for id, caption, score, tag, att, cls in zip(samples["ids"], caps["caps"], caps["scores"], tags, atts, clss):
            output.append(
                {"id": id, "caption": caption, "score": score, "tag_set1": tag, "tag_set2": tag, "attr": att, "cls": cls}
            )
        return output

    def predict_answers_memory_efficient(self, samples, *args, **kwargs):
        split_size = int(self.kwargs.get("split_size", 20))
        l_samples = len(samples["ids"])
        chunk_samples = []
        if l_samples <= split_size:
            chunk_samples = [samples]
        else:
            idxs = torch.LongTensor(range(l_samples))
            chunk_idxs = torch.split(idxs, split_size_or_sections=split_size)
            for chunk_idx in chunk_idxs:
                chunk_sample = dict()
                for key, value in samples.items():
                    if len(value) != l_samples:
                        chunk_sample[key] = value
                    elif isinstance(value, list):
                        chunk_sample[key] = [value[idx] for idx in chunk_idx]
                    else:
                        chunk_sample[key] = value[chunk_idx]
                chunk_samples.append(chunk_sample)

        output = []
        for chunk_sample in chunk_samples:
            result = self.predict_answers(chunk_sample, *args, **kwargs)
            output.extend(result)

        return output

    @classmethod
    def from_config(cls, cfg):
        model = cls(**cfg)
        if cfg.pretrained is not None:
            model.load_checkpoint(url_or_filename=cfg.pretrained)
        return model





