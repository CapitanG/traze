# Similarity Search
This project allows you to compute features and apply similarity search using a hidden layer of a ResNet50.

## Dependencies
This project depends on the project convnet2, that you can download from [here](https://github.com/jmsaavedrar/convnet2). So, you will need to set the local path of convnet2 at the top of the file [ssearch.py](ssearch.py).

## Compute Features of a Catalog
A catalog is a set of images used for querying. A catalog is defined by a text file listing all the filenames that you will process.

In addition, to make the configuration easier, a configuration file is required. This configuration file has many parameters, but you only need to pay attention to the following params:

* DATA_DIR: The directory where the data is stored. A folder named *ssearch* should exist, because it will contain all the data produced by the script.
* IMAGE_WIDTH:  Width of the input image.
* IMAGE_HEIGHT: Height of the input image.
* CHANNELS: Number of channels of the input.

You can find an example of this configuraction file in [resnet50.config](resnet50.config).

Finally, the command to compute the catalog is:
```
python ssearch.py -config resnet50.config -name RESNET -mode compute
```
where RESNET is the name of a section in the configuration file.

## Make a Query
For queryng you can use the following command:
```
python ssearch.py -config resnet50.config -name RESNET -mode search
```

As you can note, we only have changed the paramenter *mode* to *search*. After running the previous command, the sysmem will ask you for a filename, that is the input query.
```
Query: test_images/flower_1.jpg
```
For instance, you can use the test images that come with this project.  Then, the search engine will look for similar images and a collage with the results is genereted in the current folder. In this case the result is stored in the file flower_1.jpg_l2_result.png, where the first image is the query.

![aa](flower_1.jpg_l2_result.png)

# Analysis of results
In this section you will find the instructions to reproduce the graphs that help us to validate our results and draw conclusions.

We are going to reproduce the code for UMAP visualizations, clustering with Kmeans and HDBSCAN, and also the performance results of the last two clustering methods.

## UMAP visualizations
First of all download the notebook [UMAP_final_results](https://github.com/pshiguihara/USIL_Capstone_202102_G1/blob/master/visual_attributes/Analysis_results/UMAP/UMAP_final_results.ipynb) from the folder UMAP in [Analysis_results](https://github.com/pshiguihara/USIL_Capstone_202102_G1/tree/master/visual_attributes/Analysis_results) directory of this repository.

We are going to import the libraries listed at the beginning.

```
pip install umap-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn import preprocessing
import umap.umap_ as umap
```
Then for visualization with matplotlib we will create our own color dictionary.
```
custom_colors = [(255/255,0/255,0/255), #red
               (0/255, 0/255, 0/255), #black
               (0/255, 0/255, 255/255), #blue
               (0/255, 144/255, 0/255), #green
               (255/255, 255/255, 126/255), #yellow
               (144/255, 144/255, 144/255), #gray
               (172/255, 135/255, 73/255), #brown
               (255/255, 192/255, 203/255), #pink
               (149/255, 53/255, 83/255), #purple
               (255/255, 165/255, 0/255) #orange
               ]
my_cmap = LinearSegmentedColormap.from_list('UMAP_colors', custom_colors, N=10)
```

Now we will upload the output from the resnet to the notebook, assigning each one a name and converting them into dataframes.

```
from google.colab import files   
uploaded = files.upload()
color_layer1_df=pd.read_csv("data_pool1_pool_20211001-203037_Color.csv")
```

Finally we will execute the following code to achieve the visualization for each layer and attribute.
- First we prepare the dataframe

```
le = preprocessing.LabelEncoder()
feature_columns_header = ["f"+str(i) for i in range(1, color_layer1_df.shape[1])]
label_column_header = ['label']
color_layer1_df.columns = label_column_header + feature_columns_header
string_labels = ['red', 'black', 'blue', 'green', 'yellow', 'gray', 'brown', 'pink', 'purple', 'orange']
color_layer1_df['label'] = le.fit_transform(color_layer1_df['label'])
layer_data = color_layer1_df.iloc[:, 1:].values
```

- We create the UMAP object
```
reducer = umap.UMAP(random_state=42, n_neighbors=15, metric='euclidean')
embedding = reducer.fit_transform(layer_data)
```

- We create the visualization using matplotlib

```
test_colors = color_layer1_df.label
scatter_plot = plt.scatter(embedding[:, 0], embedding[:, 1], c=test_colors, cmap=my_cmap)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the color dataset layer 1', fontsize=12)
handles, labels = scatter_plot.legend_elements()
labels = string_labels
plt.legend(handles, labels)
plt.show()
```
![umap](https://user-images.githubusercontent.com/30530279/143971999-0b96e3d3-8946-4dcf-a583-4ddf18af9364.png)

## Kmeans clustering

First of all download the notebook [Kmeans_clustering_validation](https://github.com/pshiguihara/USIL_Capstone_202102_G1/blob/master/visual_attributes/Analysis_results/KMEANS-HDBSCAN/Kmeans_clustering_validation.ipynb) from the folder KMEANS-HDBSCAN in [Analysis_results](https://github.com/pshiguihara/USIL_Capstone_202102_G1/tree/master/visual_attributes/Analysis_results) directory of this repository.

We are going to import the libraries listed at the beginning.

```
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import preprocessing
```

Now we will upload the output from the resnet to the notebook, assigning each one a name and converting them into dataframes.

```
from google.colab import files   
uploaded = files.upload()
color_layer1_df=pd.read_csv("data_pool1_pool_20211001-203037_Color.csv")
texture_layer1_df=pd.read_csv("data_pool1_pool_20211001-210021_Texture.csv")
```

Then we will execute the following code to achieve the visualization of the performance metrics of each layer and attribute.

- First we create the Kmeans object

```
kmeans = KMeans(n_clusters=10)
le = preprocessing.LabelEncoder()
```

- Then we prepare the dataframe

```
feature_columns_header = ["f"+str(i) for i in range(1, color_layer1_df.shape[1])]
label_column_header = ['label']
color_layer1_df.columns = label_column_header + feature_columns_header
color_layer1_df['label']= le.fit_transform(color_layer1_df['label'])
layer_data_c1 = color_layer1_df.iloc[:, 1:].values
```

- Next we extract the true labels and those predicted by Kmeans
```
label_true_c1=color_layer1_df.iloc[:,0]
label_c1 =kmeans.fit_predict(layer_data_c1)
```

- Now we use the Adjusted mutual info score and Adjusted rand score metrics to evaluate clustering performance

```
amis_c1=adjusted_mutual_info_score(label_true_c1,label_c1)
print('amis c1: '+str(amis_c1))
ars_c1=adjusted_rand_score(label_true_c1,label_c1)
print('ars  c1: '+str(ars_c1))
```

These are the results we get:

```
amis c1: 0.5692460027654673
ars  c1: 0.3827248421526321
```

- Finally, to draw conclusions, we put the results of the metrics in a bar graph to be able to evaluate the results
```
b=['bloque 1','bloque 2','bloque 3','bloque 4','bloque 5']
x1=[ars_c1,ars_c2,ars_c3,ars_c4,ars_c5]
y1=[amis_c1,amis_c2,amis_c3,amis_c4,amis_c5]
df_color = pd.DataFrame({'Adjusted rand index':x1, 'Adjusted mutual-information score':y1},index=b)
ax_col = df_color.plot.bar(rot=0)
```
![kmeans validation](https://user-images.githubusercontent.com/30530279/143971906-0207a3d5-ef4c-411e-9df4-3ae3c141ac07.png)


## HDBSCAN clustering

First of all download the notebook [HDBSCAN_clustering_validation](https://github.com/pshiguihara/USIL_Capstone_202102_G1/blob/master/visual_attributes/Analysis_results/KMEANS-HDBSCAN/HDBSCAN_clustering_validation.ipynb) from the folder KMEANS-HDBSCAN in [Analysis_results](https://github.com/pshiguihara/USIL_Capstone_202102_G1/tree/master/visual_attributes/Analysis_results) directory of this repository.

We are going to import the libraries listed at the beginning.

```
pip install hdbscan
import hdbscan
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import preprocessing
```

Now we will upload the output from the resnet to the notebook, assigning each one a name and converting them into dataframes

```
from google.colab import files   
uploaded = files.upload()
color_layer1_df=pd.read_csv("data_pool1_pool_20211001-203037_Color.csv")
texture_layer1_df=pd.read_csv("data_pool1_pool_20211001-210021_Texture.csv")
```

Then we will execute the following code to achieve the visualization of the performance metrics of each layer and attribute

- First we create the HDBSCAN object
```
clusterer =hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, leaf_size=40)
le = preprocessing.LabelEncoder()
```

- Then we prepare the dataframe
```
feature_columns_header = ["f"+str(i) for i in range(1, color_layer1_df.shape[1])]
label_column_header = ['label']
color_layer1_df.columns = label_column_header + feature_columns_header
color_layer1_df['label']= le.fit_transform(color_layer1_df['label'])
layer_data_c1 = color_layer1_df.iloc[:, 1:].values
```

- Next we extract the true labels and those predicted by HDBSCAN to evaluate their performance
```
label_true_c1=color_layer1_df.iloc[:,0]
cluster_labels_c1=clusterer.fit_predict(layer_data_c1)
```

- Now we use the Adjusted mutual info score and Adjusted rand score metrics to evaluate clustering performance
```
amis_c1=adjusted_mutual_info_score(label_true_c1,cluster_labels_c1)
print('amis c1: '+str(amis_c1))
ars_c1=adjusted_rand_score(label_true_c1,cluster_labels_c1)
print('ars  c1: '+str(ars_c1))
```

These are the results we get:
```
amis c1: 0.31835574176306475
ars  c1: 0.09398430010300153
```

- Finally, to draw conclusions, we put the results of the metrics in a bar graph to be able to evaluate the results
```
b=['bloque 1','bloque 2','bloque 3','bloque 4','bloque 5']
x1=[ars_c1,ars_c2,ars_c3,ars_c4,ars_c5]
y1=[amis_c1,amis_c2,amis_c3,amis_c4,amis_c5]
df_color = pd.DataFrame({'Adjusted rand index':x1, 'Adjusted mutual-information score':y1},index=b)
ax_col = df_color.plot.bar(rot=0)
```
![HDBSCAN validation](https://user-images.githubusercontent.com/30530279/143971840-bb13a2fb-b331-4720-a884-c7eaac8eb49a.png)


