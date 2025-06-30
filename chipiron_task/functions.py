## PLOT VOLUMES
import matplotlib.pyplot as plt

def volume_plot(volume_rss):
    num_slices = volume_rss.shape[0]
    num_slices_to_show = 5 # You can adjust this number

    plt.figure(figsize=(15, 5)) # Adjust figure size as needed
    i=1
    for slice_index in range(1,num_slices,4):
        if slice_index >= num_slices:
            break
        print(slice_index)
        mri_slice = volume_rss[slice_index, :, :]

        plt.subplot(1, num_slices_to_show, i) # Arrange plots in a row
        plt.imshow(mri_slice, cmap='gray') # Use 'gray' colormap for MRI
        plt.title(f'Slice {slice_index}')
        plt.axis('off') # Turn off axis labels
        i += 1

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()