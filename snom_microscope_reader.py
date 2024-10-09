from PIL import Image
import os
import matplotlib.pyplot as plt

class ImageLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_list = self.load_images()
        self.images = self.load_images()
    
    def load_images(self):
        image_list = []
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(self.folder_path, filename)
                image = Image.open(image_path)
                image_list.append(image)
        return image_list

    def plot_images(self, images):
        if type(images) == list:
            num_images = len(images)
            plt.figure(figsize=(10, 6))
        
            for idx, image in enumerate(images):
                plt.subplot(1, num_images, idx + 1)
                plt.imshow(image)
                plt.title(f"Image {idx + 1}")
                plt.axis('off')

        else:
            plt.imshow(images)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    folder_path = r'C:\Users\name\data'
    loader = ImageLoader(folder_path)
    
    loaded_images = loader.load_images()
    loader.plot_images(loaded_images)


