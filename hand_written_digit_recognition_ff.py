import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import pandas as pd
import seaborn as sb
import utilities as ut
class HWDR_MNIST():
    
    def __init__(self,train_img,train_lbl,test_img,test_lbl) -> None:
        self.TRAIN_IMAGE_FILE_NAME = train_img
        self.TRAIN_LABEL_FILE_NAME = train_lbl
        self.TEST_IMAGE_FILE_NAME = test_img
        self.TEST_LABEL_FILE_NAME = test_lbl
        self.utils = ut.Utilities()
        
    def read_mnist(self):
        # Read the data from the files.
        self.train_images = idx2numpy.convert_from_file(self.TRAIN_IMAGE_FILE_NAME)
        self.train_labels = idx2numpy.convert_from_file(self.TRAIN_LABEL_FILE_NAME)

        self.test_images = idx2numpy.convert_from_file(self.TEST_IMAGE_FILE_NAME)
        self.test_labels = idx2numpy.convert_from_file(self.TEST_LABEL_FILE_NAME)

        # Normalizing Training Data
        self.x_train_raw = self.train_images.reshape(60000,784)
        mean = np.mean(self.x_train_raw)
        stddev = np.std(self.x_train_raw)
        self.x_train = (self.x_train_raw-mean) / stddev

        #Normalizing Test Data
        self.x_test_raw = self.test_images.reshape(10000,784)

        # Option 1: mean & stddev from training
        self.x_test = (self.x_test_raw - mean) / stddev

        # Option 2: mean & stddev from test
        mean_t = np.mean(self.x_test_raw)
        stddev_t = np.std(self.x_test_raw)
        self.x_test_t = (self.x_test_raw - mean_t)/stddev_t

        # One Hot Encoding Output
        self.y_train = np.zeros((60000,10))
        self.y_test = np.zeros((10000,10))

        for i, y in enumerate(self.train_labels):
            self.y_train[i][y] = 1

        for i , y in enumerate(self.test_labels):
            self.y_test[i][y] = 1
        return self.x_train, self.x_test, self.x_test_t, self.y_train, self.y_test

    def plot_pixel_dist(self,img_num):
        if len(img_num) > 3:
            print("Plotting Pixel distribution is supported only for 3 digits")
        if len(img_num) == 0:
            print("Provide atleast single image to plot pixel value distribution")
        
        figure,axes = plt.subplots(2,3,sharex=False)
        plt.title = "Distribution of Normalized vales at each Pixel (784)"
        
        w = pd.DataFrame(self.x_train_raw[img_num[0]])
        x_label = self.train_labels[img_num[0]]

        sb.violinplot(data=w,ax=axes[1,0],fill=False,color="red" ,inner="point") 
        axes[1,0].set_xlabel("digit: "+str(x_label))        
        axes[0,0].imshow(self.x_train_raw[img_num[0]].reshape(28,28),cmap='gray')

        w = pd.DataFrame(self.x_train_raw[img_num[1]])
        x_label = self.train_labels[img_num[1]]
        sb.violinplot(data=w,ax=axes[1,1],fill=False,color="green",inner="point")
        axes[1,1].set_xlabel("digit: "+str(x_label))
        axes[0,1].imshow(self.x_train_raw[img_num[1]].reshape(28,28),cmap='gray')

        w = pd.DataFrame(self.x_train_raw[img_num[2]])
        x_label = self.train_labels[img_num[2]]
        sb.violinplot(data=w,ax=axes[1,2],fill=False,color="blue",inner="point")
        axes[1,2].set_xlabel("digit: "+str(x_label))
        axes[0,2].imshow(self.x_train_raw[img_num[2]].reshape(28,28),cmap='gray')
        plt.show()

    def initialize_weights(self,tensor,biases):
        weigted_tensor = tensor
        weighted_biases = biases

        for curr_tensor in range(len(tensor)):
            temp_tensor = tensor[curr_tensor]
            rows,cols = temp_tensor.shape
            print(f"Rows: {rows} Columns: {cols}")
            for r in range(rows):
                for c in range(cols):
                    temp_tensor[r][c] = np.random.uniform(-0.1,0.1)
            weigted_tensor[curr_tensor] = temp_tensor

        for curr_bias in range(len(biases)):
            temp_bias = biases[curr_bias]
            rows,_ = temp_tensor.shape
            # print(f"Rows: {rows}")
            for r in range(rows):
                    temp_bias[r] = np.random.uniform(-0.1,0.1)
            weighted_biases[curr_bias] = temp_bias
        return weigted_tensor,biases
   
    def model_architecture(self,input_size,num_layers,num_nodes_per_layer,output):
        init_model = []
        biases = []
        # Input Layer to the First Hidden Layer
        input_weights = np.zeros((num_nodes_per_layer[0],input_size))
        init_model.append(input_weights)
        # Hidden Layers to form the network
        current_hidden_layer = 0
        while current_hidden_layer < (num_layers-1):
            hidden_weights = np.zeros((num_nodes_per_layer[current_hidden_layer+1],num_nodes_per_layer[current_hidden_layer]))
            init_model.append(hidden_weights)
            current_hidden_layer += 1
        last_layer = np.zeros((output,num_nodes_per_layer[num_layers-1]))
        init_model.append(last_layer)
       
        current_hidden_layer = 0
        while current_hidden_layer < (num_layers):
            layer_bias = np.zeros((num_nodes_per_layer[current_hidden_layer],1))
            biases.append(layer_bias)
            current_hidden_layer += 1
        last_layer_bias = np.zeros((output,1))
        biases.append(last_layer_bias)
        
        return init_model,biases

    def compute_loss(self):
        pass

    def forward_pass(self,input,weights,biases):
        # final_results = np.array((output,1))
        print("Starting Forward Pass....")
        print(input.shape)
        print(weights[0].shape)
        layer_outputs = []
        print("Start Layer 0....")
        layer_0_weights = weights[0]
        layer_results = np.dot(layer_0_weights,input) + biases[0] 
        layer_results = self.utils.tanh(layer_results)
        layer_outputs.append(layer_results)
        print(layer_results.shape)
        print("End Layer 0")
        current_layer = 1

        while current_layer < len(weights):
            print(f"Start Current Layer {current_layer}")
            print(current_layer)
            layer_weights = weights[current_layer]
            print(layer_weights.shape)
            layer_results = np.dot(layer_weights,layer_results) + biases[current_layer]
            layer_results = self.utils.tanh(layer_results)
            layer_outputs.append(layer_results)
            print(layer_results.shape)
            print(f"End Current Layer {current_layer}")
            current_layer += 1
        print("Ending Forward Pass.......")
        final_result = layer_results
        return final_result, layer_outputs
    
    def backward_pass(self):
        pass

    
    def train_model(self):
        pass



if __name__ == "__main__":
    np.random.seed(8)
    LEARNING_RATE = 0.01
    EPOCH = 20

    train_img = './/datasets//mnist//train-images.idx3-ubyte'
    train_lbl = './/datasets//mnist//train-labels.idx1-ubyte'

    test_img = './/datasets//mnist//t10k-images.idx3-ubyte'
    test_lbl = './/datasets//mnist//t10k-labels.idx1-ubyte'

    mnist = HWDR_MNIST(train_img=train_img,
                       train_lbl=train_lbl,
                       test_img=test_img,
                       test_lbl=test_lbl)

    x_train, x_test, x_test_t, y_train, y_test = mnist.read_mnist()
    
    # mnist.plot_pixel_dist([0,40,5])
    model_architecture,biases = mnist.model_architecture(784,[128,64,33,2],10)
    
    for i in range(len(biases)):
        print(f"Shape of Layer {i} Weights {model_architecture[i].shape} and bias {biases[i].shape}")

    initialized_weight_tensor,biases = mnist.initialize_weights(model_architecture,biases)
    
    for i in range(len(biases)):
        print(f"Shape of initialized layer {i} weights {initialized_weight_tensor[i].shape} and bias {biases[i].shape}")
       
    
    # X = np.ones((784,1))
    X = np.reshape(x_train[0],(784,1))
    print(f"Shape of X {X.shape}")
    final_result, layer_outputs = mnist.forward_pass(input=X,weights=initialized_weight_tensor,biases=biases)
    print(final_result.shape)
    
    utils = ut.Utilities()
    softmax_result = utils.softmax(input_vector=final_result)
    print(softmax_result)


