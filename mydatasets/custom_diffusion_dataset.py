# -*- coding:utf-8 -*-
# email:
# create: @time: 1/7/23 11:16
import PIL
import torch
import random
from tqdm.auto import tqdm
import hashlib
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from base.driver import logger
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, DiffusionPipeline

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            concepts_list,
            pretrain_model_or_path,
            tokenizer,
            device,
            size=512,
            center_crop=False,
            with_prior_preservation=False,
            num_class_images=200,
            hflip=False,
            **kwargs
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = PIL.Image.BILINEAR

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation

        for concept in concepts_list:
            if concept["instance_prompt"]:
                inst_img_path = [
                    (x, imagenet_templates_small[np.random.randint(0, len(imagenet_templates_small))].format(
                        concept["instance_prompt"])) for x in
                    Path(concept["instance_data_dir"]).iterdir() if
                    x.is_file()]
            else:
                inst_img_path = [
                    (x, concept["instance_prompt"]) for x in
                    Path(concept["instance_data_dir"]).iterdir() if
                    x.is_file()]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if not class_data_root.exists():
                    class_data_root.mkdir(parents=True, exist_ok=True)
                    class_images_path = []
                    class_prompt = []
                else:
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                if len(class_images_path) < num_class_images:
                    self.generate_with_class_prompt(pretrain_model_or_path, concepts_list, num_class_images, device)
                    class_data_root = Path(concept["class_data_dir"])
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)

        #### apply augmentation and create a valid image regions mask ####
        if np.random.randint(0, 3) < 2:
            random_scale = np.random.randint(self.size // 3, self.size + 1)
        else:
            random_scale = np.random.randint(int(1.2 * self.size), int(1.4 * self.size))

        if random_scale % 2 == 1:
            random_scale += 1

        if random_scale < 0.6 * self.size:
            add_to_caption = np.random.choice(["a far away ", "very small "])
            instance_prompt = add_to_caption + instance_prompt
            cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
            cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)

            instance_image1 = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image1 = np.array(instance_image1).astype(np.uint8)
            instance_image1 = (instance_image1 / 127.5 - 1.0).astype(np.float32)

            instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
            instance_image[cx - random_scale // 2: cx + random_scale // 2,
            cy - random_scale // 2: cy + random_scale // 2, :] = instance_image1

            mask = np.zeros((self.size // 8, self.size // 8))
            mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1,
            (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.
        elif random_scale > self.size:
            add_to_caption = np.random.choice(["zoomed in ", "close up "])
            instance_prompt = add_to_caption + instance_prompt
            cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
            cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

            instance_image = instance_image.resize((random_scale, random_scale), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            instance_image = instance_image[cx - self.size // 2: cx + self.size // 2,
                             cy - self.size // 2: cy + self.size // 2, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            if self.size is not None:
                instance_image = instance_image.resize((self.size, self.size), resample=self.interpolation)
            instance_image = np.array(instance_image).astype(np.uint8)
            instance_image = (instance_image / 127.5 - 1.0).astype(np.float32)
            mask = np.ones((self.size // 8, self.size // 8))
        ########################################################################

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

    def generate_with_class_prompt(self, pretrain_model_or_path, concepts_list, num_class_images, device):
        for i, concept in enumerate(concepts_list):
            class_images_dir = Path(concept['class_data_dir'])
            cur_class_images = len(list(class_images_dir.iterdir()))

            weight_dtype = torch.float16
            pipeline = DiffusionPipeline.from_pretrained(
                pretrain_model_or_path,
                torch_dtype=weight_dtype,
                safety_checker=None,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(concept["class_prompt"], num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset,
                                                            batch_size=1)

            pipeline.to(device)

            for example in tqdm(sample_dataloader, desc="Generating class images"):
                images = pipeline(example["prompt"], num_inference_steps=50, guidance_scale=6.,
                                  eta=1.).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class CustomDiffusionCollectFn:
    def __init__(self, tokenizer, with_prior_preservation, **kwargs):
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation

    def __call__(self, examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        mask = [example["mask"] for example in examples]
        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if self.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            mask += [example["class_mask"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        mask = torch.stack(mask)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        mask = mask.to(memory_format=torch.contiguous_format).float()

        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mask": mask.unsqueeze(1)
        }
        return batch
