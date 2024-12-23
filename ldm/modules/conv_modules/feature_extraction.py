from ldm.modules.conv_modules.modules import Feature_Block
import torch.nn as nn
import torch
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential, Downsample
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
class Feature_Extraction(TimestepBlock):
    '''
    get strcuture_cond
    '''
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 1, 2, 2),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        *args,
        **kwargs
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels #256
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4  #1024

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1),
                    normalization(model_channels)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                        Feature_Block(
                            channels=ch,
                            emb_channels=time_embed_dim,
                            dropout=dropout,
                            out_channels=mult * model_channels,
                        )
                ]
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                            )
                        )
                    )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            Feature_Block(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
            ),
            Feature_Block(
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
            )
        )

        input_block_chans.append(ch)
        self._feature_size += ch
        self.input_block_chans = input_block_chans

        self.fea_tran = nn.ModuleList([])

        for i in range(len(input_block_chans)):
            self.fea_tran.append(
                Feature_Block(
                    channels=input_block_chans[i],
                    emb_channels=time_embed_dim,
                    dropout=dropout,
                    out_channels=out_channels
                )
            )

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) #正弦编码

        result_list = []
        results = {}
        h = x.type(self.dtype)
        for module in self.input_blocks:
            last_h = h
            h = module(h, emb)
            if h.size(-1) != last_h.size(-1):
                result_list.append(last_h)
        h = self.middle_block(h, emb)
        result_list.append(h)

        assert len(result_list) == len(self.fea_tran)

        for i in range(len(result_list)):
            results[str(result_list[i].size(-1))] = self.fea_tran[i](result_list[i], emb)

        return results

if __name__ == '__main__':
    device = torch.device('cuda')
    model = Feature_Extraction(
        image_size=96,
        in_channels=4,
        model_channels=256,
        out_channels=256,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        dropout=0,
        channel_mult=(1, 1, 2, 2),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
    ).to(device)
    x = torch.rand(2,4,64,64).to(device)
    timesteps = torch.rand(2,).to(device)
    y = model(x, timesteps)
    print(y)