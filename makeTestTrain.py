import os
import shutil
from PIL import Image
import numpy as np
import random 



class Dataset:

    labelPath = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Code/30class.txt"
    validGTPath = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
    validPath = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Validation/ILSVRC2012_img_val (1)"
    testFolder = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Test"
    trainFolder = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Train"
    codeFolder = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Code"
    classes = 30
    preProcessedPath = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Pre-processed"
    noFilterPath = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/No_Filter"  
    patchesPath = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Patches"

    def getLabelNums(self):
        labelNums = []
        labelPath = self.labelPath
        with open(labelPath) as file:
            for line in file:
                words = line.split(" ")
                length = len(words)
                labelNum = int(words[length-1].strip())
                labelNums.append(labelNum)
                
        return labelNums


    def getIndices(self,labelNum):

        indices = []
        count = 0
        #validGTPath = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
        validGTPath = self.validGTPath
        with open(validGTPath) as file:
            for i in file:
                if labelNum == int(i.strip()):
                    indices.append(count)
                count+=1
        return indices

    def makeTrainTest(self):

        labelNums = self.getLabelNums()
        #labelNums = getLabelNums("/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/30class.txt")

        #validPath = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Validation/ILSVRC2012_img_val (1)"
        validPath = self.validPath
        validImages = sorted(os.listdir(self.validPath))
        validImages.remove(".DS_Store")

        testFolder = self.testFolder
        trainFolder =self.trainFolder
        #testFolder = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Test{}"
        #trainFolder = "/Users/justindwarika/Documents/NYU_2018-2019/Senior_Design/Train{}"

        count = 0
        for num in labelNums:
            indices = self.getIndices(num)
            for index in indices:
                if (count%50)//40 == 0:
                    shutil.copyfile(os.path.join(validPath,validImages[index]), os.path.join(trainFolder, os.path.basename(validImages[index])))
                else:
                    shutil.copyfile(os.path.join(validPath,validImages[index]), os.path.join(testFolder, os.path.basename(validImages[index])))
                count+=1

    def makeGTTables(self):

        codeFolder = self.codeFolder
        file = open(os.path.join(codeFolder, "gt_value_test.txt"), 'w')

        for i in range(30):
            for j in range(10):
                file.write(str(i) + "\r")
        file.close()

        file = open(os.path.join(codeFolder, "gt_value_train.txt"), 'w')

        for i in range(30):
            for j in range(40):
                file.write(str(i) + "\r")
        file.close()

    def imagePreprocess(self, img):
        width = img.size[0]
        height = img.size[1]
        Asp_ratio = width/height
        if width < height:
            new_h = int(256 * (1/Asp_ratio))
            img = img.resize((256, new_h), Image.BICUBIC)
        else:
            new_w = int(256 * Asp_ratio)
            img = img.resize((new_w, 256), Image.BICUBIC)
        
        return img
    
    def imagePreprocessNoFilter(self, img):
        width = img.size[0]
        height = img.size[1]
        Asp_ratio = width/height
        if width < height:
            new_h = int(256 * (1/Asp_ratio))
            img = img.resize((256, new_h))
        else:
            new_w = int(256 * Asp_ratio)
            img = img.resize((new_w, 256))
        return img

    def save_preprocess_image(self, imagePath):
        
        path, base = os.path.split(imagePath)
        
        image = Image.open(imagePath)
        image = self.imagePreprocess(image)
        saveFolder = self.preProcessedPath
        image.save(os.path.join(saveFolder, base))
        
        noFilter = Image.open(imagePath)
        noFilter = self.imagePreprocessNoFilter(noFilter)
        noFilterPath =self.noFilterPath
        noFilter.save(os.path.join(noFilterPath, base))
        
    def getpatches(self, image_path, n):
        img = load_image(image_path)
        img =  image_preprocess(img)
        patches=[]
        IMAGENET_MEAN = [123.68, 116.779, 103.939]
        for i in range(n):
            x = random.randint(0, img.size[0] - 224)
            y = random.randint(0, img.size[1] - 224)
            img_cropped = img.crop((x, y, x + 224, y + 224))

            cropped_im_array = np.array(img_cropped, dtype=np.float32)
        
            for j in range(3):
                cropped_im_array[:,:,j] -= IMAGENET_MEAN[j]

            patches.append(cropped_im_array)
        
        np.vstack(patches)
            
        return patches

    def savePatches(self, imagePath, n):
        path, base = os.path.split(imagePath)

        image = Image.open(imagePath).convert('RGB')
        image = self.imagePreprocess(image)
        patchesFolder = self.patchesPath
        imageNames = ["patch1.JPEG", "patch2.JPEG", "patch3.JPEG", "patch4.JPEG", "patch5.JPEG"]
        justCropNames = ["crop1.JPEG", "crop2.JPEG", "crop3.JPEG", "crop4.JPEG", "crop5.JPEG"]

        IMAGENET_MEAN = [123.68, 116.779, 103.939]
        for i in range(n):
            x = random.randint(0, image.size[0] - 224)
            y = random.randint(0, image.size[1] - 224)
            cropped = image.crop((x, y, x + 224, y + 224))

            cropped_im_array = np.array(cropped, dtype=np.float32)
            print(cropped_im_array.shape)
            cropped.save(os.path.join(patchesFolder,justCropNames[i]))
            
            for j in range(3):
                   cropped_im_array[:,:,j] -= IMAGENET_MEAN[j]
               

            patch = Image.fromarray(cropped_im_array, "RGB")
            patch.save(os.path.join(patchesFolder, imageNames[i]))
  
            
'''
def main():
    data = Dataset()
    data.makeTrainTest()

if __name__ == '__main__':
    main()
'''
    
    

    
    
            
                
                 








             
