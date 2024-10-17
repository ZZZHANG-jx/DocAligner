from re import L
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F

# class VGGPyramid(torch.nn.Module):
#     '''
#     input and target should be firstly transformed to (0,1)
#     '''
#     def __init__(self, if_train=False):
#         super(VGGPyramid, self).__init__()
#         blocks = []
#         blocks.append(models.vgg16(pretrained=True).features[:4])#1
#         blocks.append(models.vgg16(pretrained=True).features[4:9])#2
#         blocks.append(models.vgg16(pretrained=True).features[9:16]) #4
#         blocks.append(models.vgg16(pretrained=True).features[16:23])#8
#         blocks.append(models.vgg16(pretrained=True).features[23:30])#16
#         for bl in blocks:
#             for p in bl.parameters():
#                 p.requires_grad = if_train
            
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
#         self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
#     def forward(self, x,quarter_resolution_only=False, eigth_resolution=False):
#         self.device = x.device
#         x = (x-self.mean.to(self.device)) / self.std.to(self.device)
#         out = []
#         if quarter_resolution_only:
#             for i, block in enumerate(self.blocks):
#                 x = block(x)
#                 if i == 2:
#                     out.append(x)
#                     return out
#         elif eigth_resolution:
#             for i, block in enumerate(self.blocks):
#                 x = block(x)
#                 out.append(x)
#                 if i == 3:
#                     return out     
#         else:
#             for i, block in enumerate(self.blocks):
#                 x = block(x)
#                 out.append(x)
#             if float(torch.__version__[:3]) >= 1.6:
#                 x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
#             else:
#                 x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
#             out.append(x)

#             if float(torch.__version__[:3]) >= 1.6:
#                 x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
#             else:
#                 x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
#             out.append(x)                            
#             return out 

class VGGPyramid(nn.Module):
    def __init__(self, train=False):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=True)

        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False
        relu_num = 0
        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = train

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break
        self.adpmaxpool1 = nn.AdaptiveMaxPool2d(output_size=(32,32))
        self.adpmaxpool2 = nn.AdaptiveMaxPool2d(output_size=(16,16))
        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        if quarter_resolution_only:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            # outputs.append(x_full)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            outputs.append(x_half)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_' + str(3)](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                if layer_n == 1 or layer_n==2 or layer_n==3 or layer_n==5 or layer_n==6:
                    outputs.append(x)

            # if float(torch.__version__[:3]) >= 1.6:
            #     x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
            # else:
            #     x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            # outputs.append(x)

            # if float(torch.__version__[:3]) >= 1.6:
            #     x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area', recompute_scale_factor=True)
            # else:
            #     x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            # outputs.append(x)
            x = torch.nn.functional.adaptive_max_pool2d(x,(32,32))
            outputs.append(x)
            x = torch.nn.functional.adaptive_max_pool2d(x,(16,16))
            outputs.append(x)
        return outputs

