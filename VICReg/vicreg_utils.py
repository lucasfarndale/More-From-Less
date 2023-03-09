import tensorflow as tf
import matplotlib as plt
import os

def create_projector(input_shape=(2048,), output_size=8192):
    projector = tf.keras.models.Sequential()
    projector.add(tf.keras.Input(shape=input_shape))
    projector.add(tf.keras.layers.Dense(output_size))
    projector.add(tf.keras.layers.BatchNormalization())
    projector.add(tf.keras.layers.Activation('relu'))

    projector.add(tf.keras.layers.Dense(output_size))
    projector.add(tf.keras.layers.BatchNormalization())
    projector.add(tf.keras.layers.Activation('relu'))

    projector.add(tf.keras.layers.Dense(output_size))
    return projector

def create_resnet(input_shape):
    return tf.keras.applications.resnet50.ResNet50(include_top=False, weights=None, pooling='avg', input_shape=input_shape)

def create_adam_opt(lr_decayed_fn, clipnorm=1.):
    return tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn, clipnorm=clipnorm)

def plot_samples_from_ds(samples):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(samples.numpy()[n].astype(int))
        #plt.title(f"{sample_images_one_label[n]}")
        plt.axis("off")
    plt.show()
    
class UMAPCallback(tf.keras.callbacks.Callback):
    #Needs updating for e.g. variable UMAP parameters
    def __init__(self,
                 val_data,
                 show_plot=False,
                 verbose=False,
                 evaluate_clusters=False,
                 confident_embedding=True
                ):
        self.val_data = val_data
        #self.embeddings = []
        self.outputs = []
        #self.labels = []
        self.show_plot = show_plot
        self.verbose = verbose
        self.evaluate_clusters = evaluate_clusters
        self.confident_embedding = confident_embedding
        
    def on_epoch_end(self, epoch, logs=None):
        outputs = self.model.encoder_list[0].predict(self.val_data)
        reducer = cuml.UMAP(verbose=self.verbose,
                            n_neighbors=15,
                            n_components=2,
                            min_dist=0.
                           )
        embedding = reducer.fit_transform(outputs)
        labels = cuml.cluster.hdbscan.HDBSCAN(verbose=self.verbose,
                                              min_samples=100,
                                              min_cluster_size=200,
                                              cluster_selection_method='leaf'
                                             ).fit_predict(embedding)
        if self.confident_embedding:
            confident_embedding = embedding[np.where(labels!=-1)]
        #self.embeddings.append(confident_embedding)
        self.outputs.append(outputs)
        #self.labels.append(labels)
        if self.show_plot:
            if self.confident_embedding:
                plt.scatter(confident_embedding[:, 0],
                            confident_embedding[:, 1],
                            c=labels[labels!=-1],
                            s=0.1,
                            cmap="Spectral"
                            )
            else:
                plt.scatter(embedding[:, 0],
                            embedding[:, 1],
                            c=labels,
                            s=0.1,
                            cmap="Spectral"
                            )
            #wandb.log({"UMAP": wandb.Image(plt)})
            plt.show()
            plt.close()
        print(len(set(labels)))
        print(len(labels[labels!=-1])/len(labels))
        
        
def load_vicreg_weights(folder_path, enc_list, proj_list):
    for i, enc in enumerate(enc_list):
        enc.load_weights(os.path.join(folder_path,f'encoder_weights_{i}'))
    for i, proj in enumerate(proj_list):
        proj.load_weights(os.path.join(folder_path,f'projector_weights_{i}'))
        
def save_vicreg_weights(folder_path, enc_list, proj_list):
    for i, enc in enumerate(enc_list):
        enc.save_weights(os.path.join(folder_path,f'encoder_weights_{i}'))
    for i, proj in enumerate(proj_list):
        proj.save_weights(os.path.join(folder_path,f'projector_weights_{i}'))
