# TUTORIAL: Add new models to the zoo

## Model file

1. Dependencies: Add python dependencies to **setup() install_requires** in **ptimz/setup.py**
```python
setup(
    packages=find_packages(exclude=['convert', 'tests']),
    include_package_data=True,
    # install_requires=parse_requirements('requirements.txt'),
    install_requires=['numpy>=1.19.0', 'torch>=1.9.0', 'torchvision>=0.10.0', 'openmim>=0.1.5', 'timm', 'SimpleITK',
                      'pynrrd', 'pydicom', 'opencv-python',
                      'pydicom', 'Pillow', 'mmcv>=1.3.9,<=1.4.8', 'mmpose==0.24.0', 'monai', 'torchio', 'matplotlib',
                      'jupyterlab', 'jupyter_contrib_nbextensions', 'nbconvert',
                      # add dependencies and version here
                      ],
    python_requires='>=3.6',
)
```
2. Create a model file in ptimz/model_zoo, e.g. ptimz/model_zoo/litehrnet.py
3. In ptimz/model_zoo/litehrnet.py import ptimz.registry.register_model and register the new model,  
summarized as follows:
```python
from torch import nn
from .registry import register_model


# add model class and some essential modules to __all__
__all__ = ['LiteHRNet']

# pretrain configs.
default_cfgs = {
    "litehrnet_seg3d\tlung_seg": {
        'url': "ftp:// https:// pretrained weights remote url",
        'num_classes': 2,
        "input_size": (4, 128, 128, 128),
        "first_conv": 'stem.conv1.conv',
        "last_layer": 'head'
    }
}

class LiteHRNet(nn.Module):
    def __init__(self, in_chans, num_classes, num_blocks, *args, **kwargs):
        self.stem = Stem(...)
        self.layer_names = []
        for i in range(self.num_blocks):
            layer_name = f'conv{i+1}'
            self.layer_names.append(layer_name)
            layer = nn.Conv3d(...)
            self.add_module(layer_name, layer)
        self.head = nn.Linear(...)
    
    def forward(self, x):
        x = self.stem()
        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            x = layer(x)
        return self.head(x)

@register_model
def litehrnet_seg3d(pretrained=None, in_chans=None, num_classes=None, **kwargs):
    model = LiteHRNet(in_chans, num_classes, num_blocks=4)
    
    # some pretrained weights settings
    return model

```

4. Import the model in ptimz/model_zoo/__init__.py
```python
## in ptimz/model_zoo/__init__.py
from .litehrnet import *
```

Now ptimz can find the new model with the name "litehrnet_seg3d".

## in_chans and num_classes
in_chans and num_classes are key parameters in registered functions.  
&nbsp;  
**in_chans** could be:
- The first convolutional layer input channels
- nn.Linear input feature size
- nn.Embedding input feature size

If a model is used to process T1w & T2w scans, in_chans=2

**num_classes** could be:
- The last convolutional layer input channels, e.g. semantic segmentation last layer
- nn.Linear output feature size, e.g. classification last layer


## Model fine-tuning and weights transfer
ptimz users could easily fine-tune the model just giving: i) the pretrained model name, ii) input channels, iii) output channels
To make it, we need to add the following components to the model file:
### default_cfgs 
A dict exists in each model file, each key-values pair contains pretrained weights download link, model and data infos.
One architecture could be pretrained on different diseases, tissues or modalities. So keys of default_cfgs should be **"model_name\tpretrained_name"**

```python
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': 'stem.conv1.conv',
        **kwargs
    }


default_cfgs = {
    # key: model_name\tpretrained_name
    "litehrnet_seg3d\tlung_seg": {
        # Required keys
        "url": "Remote pretrained file url",
        "input_size": (input_chans, [depth,] height, width),
        "num_classes": model output channels,
        "first_conv": "stem.conv1.conv", # first layer key in model.state_dict() without suffix .weight & .bias
        "last_layer": "head_layer.final_layer.1", # last layer key in model.state_dict() without suffix .weight & .bias
        
        # other infos, help users know the details of pretrained weights
        "paper": "doi:....",
        "slice_thickness": 5.0,
        "spacing": (1.0, 1.0, 5.0),
        "input_details": "MR [T2-weighted]",
    },
    "litehrnet_seg3d\tliver_seg":{
        ...
    }
}
```

### Make the model transferable
We provide a tool ptimz.model_zoo.helpers.build_model_with_cfg to make CNN easy to fine-tune.
```python
import inspect
import logging
from .helpers import build_model_with_cfg


def _build_hrnet(pretrained_name, in_chans=1, num_classes=1000, **kwargs):
    # pretrained_name is a key in default_cfgs, or None 
    
    
    conv_cfg = dict(type=f'Conv3d')
    norm_cfg = dict(type=f'BN3d')
    head_dict = dict(nclasses=num_classes)
    extra = dict(
        head_type='segmentation',
        head_cfg=head_dict
    )

    # direct invoke
    # model = LiteHRNet(extra, in_channels=in_chans, conv_cfg=conv_cfg, norm_cfg=norm_cfg)
    # return model

    # use cfg to build
    pretrained = False if pretrained_name is False or pretrained_name is None else True
    return build_model_with_cfg(
                                # required parameters
                                model_cls=LiteHRNet, # model  
                                variant='litehrnet-base', # model variant name, no use for now
                                pretrained=pretrained, # True or False
                                default_cfg=default_cfgs.get(pretrained_name, None), 
                                num_classes=num_classes, # new output channels
                                input_channels=in_chans, # new input channels
        
                                # other model parameters
                                extra=extra, conv_cfg=conv_cfg, norm_cfg=norm_cfg)


@register_model
def litehrnet_seg3d(pretrained=None, **kwargs):
    # get this function name = "litehrnet_seg3d" 
    func_name = str(inspect.stack()[0][3])
    
    # pretrained could be True, and we should give user a default pretrained weights.
    if pretrained is True:
        pretrained = func_name + '\t' + 'lung_seg'
    elif isinstance(pretrained, str):
        pretrained = func_name + '\t' + pretrained
        if pretrained not in default_cfgs.keys():
            _logger = logging.getLogger(__name__)
            _logger.warning(f"No pretrained weights named {pretrained}, use random initialization.")
            pretrained = False

    return _build_hrnet(pretrained, 'seghead', '3d', **kwargs)
```

ptimz.model_zoo.helpers.build_model_with_cfg will:
- Download pretrained weights if not exists.
- Interpolate(reshape) model 'first_conv' input_channels to new in_chans.
- Drop the 'last_layer' weights if num_classes not equals to the pretrained weights.
- Return the instance of model_cls with model parameters


## Install or Package ptimz
Make sure the latest version of PyPA’s build installed:
```shell
pip install -U build
```

### Package ptimz
Go to the ptimz project root directory.
```shell
python -m build
```
The ptimz is packed into: **./dist/ptimz-0.0.1-py2.py3-none-any.whl**

### Install ptimz
Install by wheel
```shell
pip install -U ./dist/ptimz-0.0.1-py2.py3-none-any.whl
```

Install from source
```shell
pip install -v -e .
```
or
```shell
python setup.py develop
```
