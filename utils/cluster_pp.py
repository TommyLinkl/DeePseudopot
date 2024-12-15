import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def plot_multiple_pp(axs, calc_list, pre_or_post_anneal):
    for calc in calc_list: 
        if pre_or_post_anneal=='pre':
            PP_data = np.loadtxt(f"{calc}_inputs/init_qSpace_pot.dat")
        elif pre_or_post_anneal=='post':
            PP_data = np.loadtxt(f"{calc}_results/best_qSpace_pot.dat")
        elif pre_or_post_anneal=='optim':
            PP_data = np.loadtxt(f"{calc}_results/epoch_400_qSpace_pot.dat")
        
        axs[0,0].plot(PP_data[:,0], PP_data[:,1], label=calc.split("/")[-2])
        axs[0,1].plot(PP_data[:,0], PP_data[:,2], label=calc.split("/")[-2])
        axs[0,2].plot(PP_data[:,0], PP_data[:,3], label=calc.split("/")[-2])
        axs[1,0].plot(PP_data[:,0], PP_data[:,1], label=calc.split("/")[-2])
        axs[1,1].plot(PP_data[:,0], PP_data[:,2], label=calc.split("/")[-2])
        axs[1,2].plot(PP_data[:,0], PP_data[:,3], label=calc.split("/")[-2])

    axs[0,0].set(xlabel="q", ylabel="v(q)", title="Cs", xlim=(0,7), ylim=(-140, 20))
    axs[0,1].set(xlabel="q", ylabel="v(q)", title="I", xlim=(0,7), ylim=(-140, 20))
    axs[0,2].set(xlabel="q", ylabel="v(q)", title="Pb", xlim=(0,7), ylim=(-140, 20))
    axs[1,0].set(xlabel="q", ylabel="v(q)")
    axs[1,1].set(xlabel="q", ylabel="v(q)")
    axs[1,2].set(xlabel="q", ylabel="v(q)")
    return axs


def clustering_PP_qSpace(ax, data, num_clusters=20, mode="k-means"): 
    # Standardize the data (K-means often works better with standardized data)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Dimensionality reduction using PCA to reduce dimensions for visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    if mode=="k-means": 
        # K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(data_scaled)
        plot_title = 'K-means Clustering Visualization (PCA Reduced to 2D)'
    elif mode=="hierarchical": 
        # Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
        clusters = hierarchical.fit_predict(data_scaled)
        plot_title = 'Hierarchical Clustering Visualization (PCA Reduced to 2D)'
    else: 
        raise NotImplementedError("The clustering mode is not implemented. ")

    # Visualize the clustering in 2D
    scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='tab20', s=50)
    ax.set(title=plot_title, xlabel='Principal Component 1', ylabel='Principal Component 2')
    plt.colorbar(scatter, ax=ax, label='Cluster Label')
    
    return ax, clusters


#################################################
if __name__ == "__main__":
    calc_list = []
    for i in range(1, 101):
        calc_list.append(f"CALCS/CsPbI3_ultraSmall_round6_longerQ/optim_{i}")
    figName = "CALCS/CsPbI3_ultraSmall_round6_longerQ/cluster_optim/PCA_clusters.pdf"
    figPrefix_indivCluster = "CALCS/CsPbI3_ultraSmall_round6_longerQ/cluster_optim/cluster"
    num_clusters = 30
    pre_or_post_anneal = 'optim'   # 'post'

    data_list = []
    data_names = []
    for calc in calc_list:
        file_path = f"{calc}_results/epoch_400_qSpace_pot.dat"

        if os.path.exists(file_path):
            try:
                this_PP = np.loadtxt(file_path)
                concatenated_data = np.concatenate((this_PP[:958, 1], this_PP[:958, 2], this_PP[:958, 3]))
                data_list.append(concatenated_data)
                data_names.append(calc)
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
        else:
            print(f"File {file_path} does not exist.")

    data = np.vstack(data_list)
    # print(f"Shape of concatenated data: {data.shape}")
    # print(f"Length of the name list: {len(data_names)}")

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax, clusters = clustering_PP_qSpace(ax, data, num_clusters, "hierarchical")
    fig.tight_layout()
    fig.savefig(figName)

    cluster_dict = {i: [] for i in range(num_clusters)}
    for idx, cluster_label in enumerate(clusters):
        cluster_dict[cluster_label].append(data_names[idx])
    for cluster_label, names in cluster_dict.items():
        cluster_cost = []
        min_cost = float('inf')
        best_pick = None

        for name in names:
            if pre_or_post_anneal == 'pre': 
                cost_file_path = f"{name}_inputs/setup_init_cost.dat"
            elif pre_or_post_anneal == 'post': 
                cost_file_path = f"{name}_results/final_mc_cost.dat"
            elif pre_or_post_anneal == 'optim': 
                cost_file_path = f"{name}_results/final_training_cost.dat"
            
            if os.path.exists(cost_file_path):
                try:
                    if pre_or_post_anneal == 'pre': 
                        cost_value = np.loadtxt(cost_file_path)
                    elif pre_or_post_anneal == 'post': 
                        cost_value = np.loadtxt(cost_file_path)[-1, 3]
                    elif pre_or_post_anneal == 'optim': 
                        cost_value = np.loadtxt(cost_file_path)[-1, 1]
                    cluster_cost.append(cost_value)
                    if cost_value < min_cost:
                        min_cost = cost_value
                        best_pick = name
                except Exception as e:
                    print(f"Could not read {cost_file_path}: {e}")
            else:
                print(f"File {cost_file_path} does not exist.")
        
        # Print the cluster info with the best pick
        # print(f"Cluster {cluster_label+1}: Best Pick: {best_pick.split('/')[-2]} (Cost: {min_cost})")
        # print(f"Cluster cost: {cluster_cost}")
        # print(f"{', '.join(name.split('/')[-2] for name in names)}\n")
        print(f"Cluster {cluster_label+1}: Best Pick: {best_pick} (Cost: {min_cost})")
        print(f"Cluster cost: {cluster_cost}")
        print(f"{', '.join(name for name in names)}\n")

        fig, axs = plt.subplots(2, 3, figsize=(9, 6))
        axs = plot_multiple_pp(axs, names, pre_or_post_anneal)
        fig.tight_layout()
        fig.savefig(f"{figPrefix_indivCluster}_{cluster_label+1}.pdf")
        plt.close('all')
