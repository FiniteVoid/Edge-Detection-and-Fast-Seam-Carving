from sourceCodes.q2Scripted import *
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid")
images = ["Media/castle.jpg", "Media/York.jpg"]
castleConfig = {
    "resizeLongEdgeTo": 512,
    "CarvedWidth": 400,
    "CarvedHeight": 300
}
yorkConfig = {
    "resizeLongEdgeTo": 512,
    "CarvedWidth": 400,
    "CarvedHeight": 300
}
configs = [castleConfig, yorkConfig]


def solution(quality=100):
    step = 100-quality+1
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))  # 3 rows, 3 columns
    plt.axis('off')
    for i in range(len(images)):
        config = configs[i]
        img = imageRead(images[i])
        img = resizeLongEdge(img, config["resizeLongEdgeTo"])
        tensor = img2tensor(img)
        # Original Image

        ax[i, 0].set_title(f"Original Image ({img.shape[1]} X {img.shape[0]})")
        ax[i, 0].imshow(img, cmap='gray')

        # Carved Image
        carvedImg = MySeamCarving2(
            tensor, config["CarvedWidth"], config["CarvedHeight"], step)

        ax[i, 1].set_title(
            f"Seam Carved Image ({carvedImg.shape[3]} X {carvedImg.shape[2]})")
        ax[i, 1].imshow(tensor2img(carvedImg), cmap='gray')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


if __name__ == '__main__':
    print("Running Solution. Please wait...")
    quality = 80    # The lower the faster (<80 not recommended)
    solution(quality)
