import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import os

def genMovie(resultsFolder, writeMovieFile, numEpochs):
    image_files = [f'{resultsFolder}epoch_{i}_plotBS.png' for i in range(1, numEpochs+1)]

    images = []
    for image_file in image_files:
        if os.path.exists(image_file):
            img = Image.open(image_file)
            images.append(img)
        else:
            break

    print("\nCreating a figure")
    fig = plt.figure(figsize=(9,4))

    # Function to update the frame
    def update_frame(frame_number):
        plt.imshow(images[frame_number])
        plt.axis('off')  # Turn off axis

    print("Creating the animation")
    ani = animation.FuncAnimation(fig, update_frame, frames=len(images), repeat=False)

    ani.save(writeMovieFile, writer='ffmpeg', fps=8)

    # Show the animation
    # plt.show()
    plt.close()



'''
calcList = ['results_64kpts_CBM',
'results_64kpts_CB',
'results_64kpts_CBVBM',
'results_64kpts_CBVB',

'results_extrema6_CBM',
'results_extrema6_CB',
'results_extrema6_CBVBM',
'results_extrema6_CBVB',

'results_RG_CBVBM',
'results_RG_CBVB',

'r2_results_64kpts_CBM',
'r2_results_64kpts_CB',
'r2_results_64kpts_CBVBM',
'r2_results_64kpts_CBVB',

'r2_results_extrema6_CBVB',
'r2_results_extrema6_CBVBM',
'r2_results_extrema6_CB',
'r2_results_extrema6_CBM',

'r2_results_RG_CBVBM',
'r2_results_RG_CBVB']
'''

calcName = 'results_64kpts_CBM_0'
for calcName in calcList: 
    for lr in [4]: # range(5): 
        resultsFolder = f'CALCS/CsPbI3_manualReorder/{calcName}_{lr}/'
        writeMovieFile = f'CALCS/CsPbI3_manualReorder/movies/{calcName}_{lr}.mp4'
        print(resultsFolder)
        numEpochs = 200
        genMovie(resultsFolder, writeMovieFile, numEpochs)