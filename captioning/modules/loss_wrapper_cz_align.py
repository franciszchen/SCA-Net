# 与luo完全相同
# 新增些注释，强调不用self-critical, 不用label smooth

import torch
import torch.nn as nn
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward



class LanguageModelCriterion(nn.Module):
    """
    ############# cz found using this choice 
    """
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, reduction='mean'):
        # print("input shape: ", input.shape) # torch.Size([9, 17, 35]) # 最后一维是分类
        # print("target shape: ", target.shape) # torch.Size([9, 1, 17])
        # print("mask shape: ", mask.shape) # torch.Size([9, 1, 17])
        # print("reduction: ", reduction) # mean
        # print("input: \n", input[0])
        # print("target: \n", target[0])
        # print("mask: \n", mask[0])
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        N,L = input.shape[:2] # N=9, L=17
        # truncate to the same size
        target = target[:, :input.size(1)]
        # print("before mask:\n", mask)
        mask = mask[:, :input.size(1)].to(input)
        # print("after mask:\n", mask) # 没啥区别

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask # torch.Size([9, 17]), 估计是每一位预测的CE loss
        # print("output: \n", output)
        # print(output.shape)

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            # cz found using this choice
            output = torch.sum(output) / torch.sum(mask)

        return output


class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        if opt.label_smoothing > 0:
            print("crit #1")
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            # print("crit #2")
            ############# cz found using this choice ！！！
            # ************************************** #
            # my: enter this step
            # print('Using LanguageModelCriterion')
            self.crit = LanguageModelCriterion()
            # self.crit = losses.LanguageModelCriterion()
            # ************************************** #

        self.rl_crit = losses.RewardCriterion() # 未使用
        self.struc_crit = losses.StructureLosses(opt) # 未使用

    def forward(self, 
                # prototypes,
                prototypes_text, 
                prototypes_visual,  
                fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, drop_worst_flag):
        """
        model_out = dp_lw_model(
            /fc_feats, att_feats, labels, masks, /att_masks, 
            data['gts'], -> gts  跟labels类似，舍弃第一位0
            torch.arange(0, len(data['gts'])), -> gt_indices 没啥用, batch内的样本idx, tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], device='cuda:0')
            sc_flag, <- False
            struc_flag,  <- False
            drop_worst_flag <- False
        )
        """
        opt = self.opt
        
        out = {}

        reduction = 'none' if drop_worst_flag else 'mean' # mean
        if struc_flag:
            print("Loss Warpper #1")
            if opt.structure_loss_weight < 1:
                print("Loss Warpper #1-1")
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:], reduction=reduction)
            else:
                print("Loss Warpper #1-2")
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if opt.structure_loss_weight > 0:
                print("Loss Warpper #1-3")
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                            or not 'margin' in opt.structure_loss_type,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts, reduction=reduction)
            else:
                print("Loss Warpper #1-4")
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']

        # ************************************** #
        elif not sc_flag:
            # print("Loss Warpper #2")
            ############# cz found using this choice ！！！
            model_output, (feature_decoder, feature_encoder, seq), (out_encoder, out_text), (visual_concept_predict, text_concept_predict) = self.model(
                # prototypes,
                prototypes_text, 
                prototypes_visual, 
                fc_feats, att_feats, labels[..., :-1], att_masks
            )
            loss = self.crit(
                model_output, 
                labels[..., 1:], 
                masks[..., 1:], 
                reduction=reduction # mean
            )
            out['model_output'] = model_output
            out['feature_decoder'] = feature_decoder # 无用
            out['feature_encoder'] = feature_encoder # 无用
            out['label'] = seq
            # out['predict_prototype'] = predict_prototype
            out['mid_token_visual'] = out_encoder #
            out['mid_token_text'] = out_text
            out['visual_concept_predict'] = visual_concept_predict
            out['text_concept_predict'] = text_concept_predict
            
            # out['seq'] = seq
            # loss = self.crit(
            #     self.model(fc_feats, att_feats, labels[..., :-1], att_masks), 
            #     labels[..., 1:], 
            #     masks[..., 1:], 
            #     reduction=reduction # mean
            # )
            # 输出out仅包含'loss'
        # ************************************** #

        else:
            # print("Loss Warpper #3")
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                    mode='sample',
                    opt={'sample_method': opt.sc_sample_method,
                         'beam_size': opt.sc_beam_size})
            self.model.train()
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                    opt={'sample_method':opt.train_sample_method,
                        'beam_size':opt.train_beam_size,
                        'sample_n': opt.train_sample_n},
                    mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward, reduction=reduction)
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        
        # print("loss: ", loss) # 4.7251
        # print("model_output: ", model_output.shape) # torch.Size([9, 17, 35])
        # print("feature: ", feature.shape) # torch.Size([9, 17, 512])
        # print("seq: ", seq.shape) # torch.Size([9, 17])
        # print("seq: ", seq)

        return out
