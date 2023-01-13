# Customize Your Diffusion
![demo](gallery/demo.png)

This is a collection of popular dreambooth-like/text-inversion-like methods. 

## How to use
1. downloaded pretrained stable-diffusion models you want to use
2. prepare your image
3. copy and paste the path to the image folder into the configuration file. 

The file `dream_booth_base.yaml` serves as an example and automatically overrides `base.yaml`. Unneeded configurations can be omitted for a more concise and friendly editing.

4. train !

**Dreambooth**
```python
python mytools/train_experiment.py --config_file config/custom_diffusion/dream_booth_base.yaml
```
**CustomDiffusion**
```python
python mytools/train_experiment.py --config_file config/custom_diffusion/custom_diffusion_base.yaml
```
## How to train a satisfying dreambooth model
Fine-tuning parameters matter. To avoid overfitting, you need to pay close attention to the number of fine-tuning steps and the learning rate when using a text encoder. A recommended approach is to use 5-10 square images and train the model for 1500 steps with a learning rate of 1e-6, with 500 steps only training the text encoder and 1000 steps for the UNet alone. Additionally, it is important to use square images as input to avoid distortions caused by image resizing.

My Recipe:

- lr: 1e-6
- 500 steps text encoder + unet and 1000 steps only unet
- at least 4 square images

<hr/>

As for custom diffusion or any other methods that tries to distill images into an embedding, the learning should be large enough to get a satisfying result. Here is a recipe I used for custom diffusion

- lr: 4.0e-05 


## Reference
    @article{ruiz2022dreambooth,
      title={Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation},
      author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
      journal={arXiv preprint arXiv:2208.12242},
      year={2022}
    }

    @article{gal2022image,
      title={An image is worth one word: Personalizing text-to-image generation using textual inversion},
      author={Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H and Chechik, Gal and Cohen-Or, Daniel},
      journal={arXiv preprint arXiv:2208.01618},
      year={2022}
    }

    @article{kumari2022multi,
      title={Multi-Concept Customization of Text-to-Image Diffusion},
      author={Kumari, Nupur and Zhang, Bingliang and Zhang, Richard and Shechtman, Eli and Zhu, Jun-Yan},
      journal={arXiv preprint arXiv:2212.04488},
      year={2022}
    }