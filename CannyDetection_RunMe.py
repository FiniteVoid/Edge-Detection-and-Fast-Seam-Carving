from sourceCodes.q1Scripted import *
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid")

fruitConfig = {
    "sigma": 1.75,
    "lowThreshold": 0.0104,
    "highThreshold": 0.0154
}
fvConfig = {
    "sigma": 1.75,
    "lowThreshold": 0.0314,
    "highThreshold": 0.0474
}
coreConfig = {
    "sigma": 1.75,
    "lowThreshold": 0.0114,
    "highThreshold": 0.0194
}
images = ["Media/bowl-of-fruit.jpg", "Media/castle.jpg", "Media/core.jpg"]
configs = [fruitConfig, fvConfig, coreConfig]


def solution():
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))  # 3 rows, 3 columns
    plt.axis('off')
    prog = 0
    for i in range(len(images)):
        config = configs[i]
        img = imageRead(images[i])
        tensor = img2tensor(img)
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')
        # Original Image
        if (i == 0):
            ax[i, 0].set_title(f"{i+1}. Original Image:")
        ax[i, 0].imshow(img, cmap='gray')
        prog += 1
        progress = (prog / 9) * 100
        sys.stdout.write("\rProgress: %d%%" % progress)
        sys.stdout.flush()

        # NMS Image
        nms_img = MyCanny(tensor, config["sigma"], config["highThreshold"])
        if (i == 0):
            ax[i, 1].set_title(f"{i+1}. NMS Edge:")
        ax[i, 1].imshow(tensor2img(nms_img), cmap='gray')
        # plt.imsave(images[i] + "_NMS.jpg",
        #            np.squeeze(tensor2img(nms_img)), cmap='gray')
        prog += 1
        progress = (prog / 9) * 100
        sys.stdout.write("\rProgress: %d%%" % progress)
        sys.stdout.flush()

        # Hysteresis Image
        hyst_img = MyCannyFull(tensor, config["sigma"],
                               config["lowThreshold"], config["highThreshold"])
        if (i == 0):
            ax[i, 2].set_title(f"{i+1}. Hysteresis Reconstruction:")
        ax[i, 2].imshow(tensor2img(hyst_img), cmap='gray')
        # plt.imsave(images[i] + "_Hyst.jpg",
        #            np.squeeze(tensor2img(hyst_img)), cmap='gray')
        prog += 1
        progress = (prog / 9) * 100
        sys.stdout.write("\rProgress: %d%%" % progress)
        sys.stdout.flush()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.95)
    plt.show()


if __name__ == '__main__':
    print("Running Combined Solution for 1.1, 1.2 (Takes around 20-40 seconds). Please wait...")
    solution()
