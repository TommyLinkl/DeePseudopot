import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os

def genMovie(resultsFolder, writeMovieFile, numEpochs=9999, preEpochs=True, mc=True, type='BS'):
    initImage = []
    if os.path.exists(f'{resultsFolder}initZunger_plot{type}.png'):
        img = Image.open(f'{resultsFolder}initZunger_plot{type}.png')
        initImage.append(img)

    preImages = []
    if preEpochs: 
        preImage_files = [f'{resultsFolder}preEpoch_{i}_plot{type}.png' for i in range(1, 9999)]
        for image_file in preImage_files:
            if os.path.exists(image_file):
                img = Image.open(image_file)
                preImages.append(img)
            # else:
                # break

    mcImages = []
    if mc: 
        # mcImage_files = [f'{resultsFolder}mc_iter_{i}_plot{type}.png' for i in range(0, 9999)]
        mcImage_files = [f'{resultsFolder}mc_iter_{i}_plot{type}.png' for i in range(0, 9999, 50)]
        for image_file in mcImage_files:
            if os.path.exists(image_file):
                img = Image.open(image_file)
                mcImages.append(img)
            # else:
                # break

    image_files = [f'{resultsFolder}epoch_{i}_plot{type}.png' for i in range(1, numEpochs+1)]
    images = []
    for image_file in image_files:
        if os.path.exists(image_file):
            img = Image.open(image_file)
            images.append(img)
        # else:
            # break

    images = initImage + preImages + mcImages + images
    print(f"{len(images)} frames in total. ")

    fig = plt.figure(figsize=(9,4))

    # Function to update the frame
    def update_frame(frame_number):
        plt.imshow(images[frame_number])
        plt.axis('off')  # Turn off axis

    print("Creating the animation")
    ani = animation.FuncAnimation(fig, update_frame, frames=len(images), repeat=False)

    FFwriter = animation.FFMpegWriter(fps=8,
                                      extra_args=['-vcodec', 'libvpx-vp9', '-b:v', '2M', '-crf', '20'])
    ani.save(writeMovieFile, writer=FFwriter)
    # ani.save(writeMovieFile, writer='ffmpeg', codec='libvpx-vp9', fps=8)

    # plt.show()
    plt.close()
