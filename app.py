import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from geopy.distance import geodesic
from sklearn.preprocessing import OneHotEncoder

# --- Define the GCN Model ---
class PathOptimizerGCN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.edge_scorer = nn.Sequential(
            nn.Linear(2*hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        src, dst = edge_index
        src_features = x[src]
        dst_features = x[dst]
        edge_repr = torch.cat([src_features, dst_features, edge_attr], dim=1)
        edge_scores = torch.sigmoid(self.edge_scorer(edge_repr))
        return edge_scores.squeeze()

# Load the saved model
model_path = "gcn_model.pth"
model = PathOptimizerGCN(node_dim=3, edge_dim=4, hidden_dim=64)
model.load_state_dict(torch.load(model_path))
model.eval()

# --- Streamlit UI ---
st.title("Network Path Optimization with GCN")
uploaded_file = st.file_uploader("Upload a GraphML file", type=["graphml"])

if uploaded_file:
    # Read GraphML file
    G = nx.read_graphml(uploaded_file)
    G = G.to_undirected()

    # Define label mapping
    label_mapping = {
        'High Bandwidth': 'High',
        'Medium-High Bandwidth': 'Medium',
        'Low Bandwidth': 'Low',
        'Medium-Low Bandwidth': 'Low'
    }

    # Apply label mapping before collecting unique labels
    for u, v, data in G.edges(data=True):
        data['LinkLabel'] = label_mapping.get(data.get('LinkLabel', 'Low'), 'Low')

    # Collect unique transformed labels for fitting OneHotEncoder
    unique_labels = list(set(data['LinkLabel'] for _, _, data in G.edges(data=True)))
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(np.array(unique_labels).reshape(-1, 1))  # Fit on transformed labels

    # Transform labels for all edges
    for u, v, data in G.edges(data=True):
        data['OneHotLinkLabel'] = encoder.transform([[data['LinkLabel']]]).tolist()[0]
    
    # Compute distances
    for u, v, data in G.edges(data=True):
        coords_u = (data.get('Latitude', 0), data.get('Longitude', 0))
        coords_v = (data.get('Latitude', 0), data.get('Longitude', 0))
        data['distance'] = geodesic(coords_u, coords_v).km

    # Prepare PyG data
    node_features = []
    for _, data in G.nodes(data=True):
        if 'Latitude' in data and 'Longitude' in data and 'Internal' in data:
            node_features.append([data['Latitude'], data['Longitude'], data['Internal']])
        else:
            node_features.append([0, 0, data.get('Internal', 0)])
    node_features = np.array(node_features)
    node_features = torch.tensor(node_features, dtype=torch.float)
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v, data in G.edges(data=True)]).t().contiguous()
    reverse_edges = edge_index[[1, 0]]
    edge_index = torch.cat([edge_index, reverse_edges], dim=1)
    edge_attr = torch.tensor([data['OneHotLinkLabel'] + [data['distance']] for _, _, data in G.edges(data=True)], dtype=torch.float)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    # Get edge scores
    with torch.no_grad():
        edge_scores = model(data)
    
    # Display network
    st.subheader("Network Visualization")
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500)
    st.pyplot(plt)
    
    # Find paths
    st.subheader("Find Optimal Paths")
    source = st.text_input("Enter Source Node")
    target = st.text_input("Enter Target Node")
    
# IF I WANT TO USE LOW FOR RELIABILITY
# def get_path_reliability(G, path):
#     """Calculate path reliability based on bandwidth labels"""
#     bandwidth_levels = {'Low': 0, 'Medium': 1, 'High': 2}
#     min_bandwidth = 2  # Start with highest possible (High)

#     for i in range(len(path) - 1):
#         u, v = path[i], path[i + 1]

#         # Handle undirected graph
#         if G.has_edge(u, v):
#             data = G[u][v]
#         elif G.has_edge(v, u):
#             data = G[v][u]
#         else:
#             return ('Unknown', '⚠️')  # No valid edge found

#         # Extract the first available 'LinkLabel' from the nested dictionary
#         bandwidth = 'Low'  # Default if missing
#         for key, value in data.items():
#             if isinstance(value, dict) and 'LinkLabel' in value:
#                 bandwidth = value['LinkLabel']
#                 break  # Stop once we find a valid label
        
#         # Update min bandwidth value
#         min_bandwidth = min(min_bandwidth, bandwidth_levels.get(bandwidth, 0))

#     # Map numerical values to reliability labels
#     reliability_map = {
#         0: ('Low', '🔴'),
#         1: ('Medium', '🟡'),
#         2: ('High', '🟢')
#     }
#     return reliability_map[min_bandwidth]


def get_path_reliability(G, path):
    """Determine the overall path reliability based on the strongest bandwidth link."""
    bandwidth_levels = {'Low': 0, 'Medium': 1, 'High': 2}
    max_bandwidth = 0  # Start with lowest possible

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        # Ensure edge exists (handle undirected case)
        if G.has_edge(u, v):
            data = G[u][v]  
        elif G.has_edge(v, u):
            data = G[v][u]  
        else:
            return ('Unknown', '⚠️')  # No valid edge found

        # Extract the first available 'LinkLabel' from the nested dictionary
        bandwidth = 'Low'  # Default if missing
        for key, value in data.items():
            if isinstance(value, dict) and 'LinkLabel' in value:
                bandwidth = value['LinkLabel']
                break  # Stop once we find a valid label
        
        # Update max bandwidth value
        max_bandwidth = max(max_bandwidth, bandwidth_levels.get(bandwidth, 0))

    # Map numerical values to reliability labels
    reliability_map = {
        0: ('Low', '🔴'),
        1: ('Medium', '🟡'),
        2: ('High', '🟢')
    }
    return reliability_map[max_bandwidth]


def print_path_bandwidth(G, path):
    """Print the bandwidth value for each edge in the path."""
    st.write(f"### Bandwidth values for path: {' → '.join(map(str, path))}")

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        # Check both directions for undirected graphs
        if G.has_edge(u, v):
            data = G[u][v]
        elif G.has_edge(v, u):
            data = G[v][u]
        else:
            st.write(f"⚠️ Edge ({u}-{v}) not found!")
            continue

        # Print all attributes for debugging
        #st.write(f"Edge ({u}-{v}) - Attributes: {data}")

        # Extract the first available 'LinkLabel' from the nested dictionary
        bandwidth = "Missing"
        for key, value in data.items():
            if isinstance(value, dict) and 'LinkLabel' in value:
                bandwidth = value['LinkLabel']
                break

        st.write(f"Edge ({u}-{v}) - Bandwidth: **{bandwidth}**")


if st.button("Find Path"):
    try:
        # Create scored graph with proper edge indexing
        nodes = list(G.nodes())
        scored_graph = nx.Graph()
        
        # Add edges with scores using correct indices
        for i in range(data.edge_index.size(1)):
            src_idx = data.edge_index[0, i].item()
            dst_idx = data.edge_index[1, i].item()
            u = nodes[src_idx]
            v = nodes[dst_idx]
            score = edge_scores[i].item()
            scored_graph.add_edge(u, v, weight=1-score)
        
        # Find paths
        paths = list(nx.shortest_simple_paths(scored_graph, source, target, weight='weight'))
        st.write("## Optimal Paths")
        
        # Display paths with reliability and country information
        for i, path in enumerate(paths[:3]):
            reliability, emoji = get_path_reliability(G, path)
            
            # Print bandwidth values for this path
            print_path_bandwidth(G, path)

            # Get country path
            country_path = []
            for node in path:
                country = G.nodes[node].get('Country', 'Unknown')
                country_path.append(f"{country} ({node})")
            
            # Display path details
            st.markdown(
                f"""
                **Path {i+1}** {emoji}  
                Reliability: **{reliability}**  
                Nodes: `{' → '.join(map(str, path))}`  
                Countries: {' → '.join(country_path)}
                """
            )
            
    except nx.NetworkXNoPath:
        st.error("No path exists between the selected nodes!")
    except nx.NodeNotFound:
        st.error("Invalid node selected! Please check the node IDs.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

