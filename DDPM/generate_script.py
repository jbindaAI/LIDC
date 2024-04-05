from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt




def generate(n_sample, guide_w, path_to_model, test_gen=True, grid_name=None):
    device = torch.device("cuda")
    ddpm = torch.load(path_to_model)
    ddpm = ddpm.to(device)   
    ddpm.eval()
    
    with torch.no_grad():
        x_gen, c, x_gen_store = ddpm.sample(n_sample, (1, 32, 32), device, guide_w=guide_w)
    if test_gen:
        plot = plot_grid(x_gen, n_sample, c)
        figure = plot.get_figure()
    else:
        return x_gen, c, x_gen_store 


def generate_fake_dataset(model_name, 
                          dir_to_save_imgs, 
                          n_sample,
                          max_guidance,
                          min_guidance,
                          guidance_steps
                         ):
    ## Path to model:
    path_to_model = f"/home/dzban112/DDPM_repo/saved_model/{model_name}.pth"

    # Guidance range and number of samples per step:
    guidance_range = np.linspace(min_guidance, max_guidance, guidance_steps)
    samples_per_step = n_sample//guidance_steps
    if samples_per_step % 2 != 0:
        samples_per_step += 1
        print(f"From a technical reason, model will generate {samples_per_step * len(guidance_range)} images.")

    # GENERATION:
    id = 1
    for guidance in guidance_range:
        x_gen, c, _ = generate(n_sample=samples_per_step, guide_w=guidance, path_to_model=path_to_model, test_gen=False)
        assert x_gen.shape[0] == c.shape[0], "Unconsistent shapes!"
        ## Saving:
        ## Paths:
        class_0_storage_folder = f"/home/dzban112/DDPM_repo/gen_imgs/{dir_to_save_imgs}/guidance_{guidance:.3f}/class0/" 
        class_1_storage_folder = f"/home/dzban112/DDPM_repo/gen_imgs/{dir_to_save_imgs}/guidance_{guidance:.3f}/class1/"
        Path(class_0_storage_folder).mkdir(parents=True, exist_ok=True)
        Path(class_1_storage_folder).mkdir(parents=True, exist_ok=True)
        ## Loop
        for i in range(x_gen.shape[0]):
            if c[i] == 0:
                save_path = class_0_storage_folder
            elif c[i] == 1:
                save_path = class_1_storage_folder
            img = x_gen[i].cpu()
            name = str(id).zfill(4)
            torch.save(img.clone(), f"{save_path}/{name}.pt")
            id+=1


if __name__ == '__main__':
    # Below: generating 200 imgs for each guidance level (200*10=2000)
    args = {
    "model_name": "model_256F_1000T_34",
    "dir_to_save_imgs": "gen_4000_0.889",
    "n_sample": 2000,
    "max_guidance": 2.000,
    "min_guidance": 0.000,
    "guidance_steps": 10
    }
    generate_fake_dataset(**args)