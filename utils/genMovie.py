import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os

def genMovie(resultsFolder, writeMovieFile, numEpochs=9999, preEpochs=True):
    initImage = []
    if os.path.exists(f'{resultsFolder}initZunger_plotBS.png'):
        img = Image.open(f'{resultsFolder}initZunger_plotBS.png')
        initImage.append(img)

    preImages = []
    if preEpochs: 
        preImage_files = [f'{resultsFolder}preEpoch_{i}_plotBS.png' for i in range(1, 9999)]
        for image_file in preImage_files:
            if os.path.exists(image_file):
                img = Image.open(image_file)
                preImages.append(img)
            else:
                break

    image_files = [f'{resultsFolder}epoch_{i}_plotBS.png' for i in range(1, numEpochs+1)]
    images = []
    for image_file in image_files:
        if os.path.exists(image_file):
            img = Image.open(image_file)
            images.append(img)
        else:
            break

    images = initImage + preImages + images
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

'''
excluded_calcName = 'r3_results_preSGD_64kpts_CBVB_0'
calcList = [f'r3_results_{opt}_64kpts_{weight}_{lr}'
            for opt in ['adam', 'preAdam', 'preSGD']
            for weight in ['CBM', 'CB', 'CBVBM', 'CBVB']
            for lr in range(5)
            if f'r3_results_{opt}_64kpts_{weight}_{lr}' != excluded_calcName]

calcList = [f'r3_results_{opt}_64kpts_{weight}_{lr}'
            for opt in ['preSGD']
            for weight in ['CBVBM', 'CBVB']
            for lr in range(5)
            if f'r3_results_{opt}_64kpts_{weight}_{lr}' != excluded_calcName]

calcList = [f'r3_results_preAdam_64kpts_{weight}_{lr}'
            for weight in ['CBM', 'CB', 'CBVBM', 'CBVB']
            for lr in range(5)]

calcList = [f'r4_results_adam_64kpts_{weight}_{lr}'
            for weight in ['CBM', 'CB', 'CBVBM', 'CBVB']
            for lr in range(2,5)]

calcList = ['r4_results_adam_64kpts_CB_4']
print(calcList)

for calcName in calcList: 
    resultsFolder = f'CALCS/CsPbI3_manualReorder/{calcName}/'
    writeMovieFile = f'CALCS/CsPbI3_manualReorder/movies/{calcName}.mp4'
    print(resultsFolder)
    numEpochs = 200
    genMovie(resultsFolder, writeMovieFile, numEpochs)'''