# GAT

注意力层：

输入：节点特征，节点为N，每个节点的特征为F

输出：新节点特征：节点为N，每个节点的特征为$F^{'}$

四个部分：

1. 线性变换：为了获得足够的表达能力将输入特征转换为更高级别的特征。

$$
z_i^{l}=W^{l}h_i^{l} \\
W:共享参数；
h_i:节点特征
$$

2. 注意力系数：

$$
e_{ij}^{l}=LeakyReLU(\vec{a}^{T}(z_i^{l}||z_j^{l})) \\
\vec{a}:注意力参数 \\
e_{ij}^{l}：节点j对节点i的中重要性
$$

3. Softmax：归一化处理：

$$
\alpha_{ij}^{l}=\frac{exp(e_{ij}^{l})}{ {\textstyle \sum_{k\subset N(i)}^{}} exp(e_{ik}^{l})} 
$$

4. 聚合：更新节点特征

$$
h_i^{l+1}=\sigma (\sum_{j\subset N(i)}^{} )a_{ij}^{l}z_{j}^{l}
$$

## 数据集：Cora

2708个节点（一篇论文），被分为8个类别，特征1433维(元素0 or 1)。若两篇论文有引用关系，则连通；

文件格式

* cora.content：所有论文的独自的信息，一共2708行(节点)，每一行由三部分组成：	
  * 论文编号；
  * 特征：1433
  * 类别：文本形式给出；
* cora.cites：5429行，每一行有两个论文编号，表示第二个编号论文引用第一个编号的论文；

## pyG

安装

```bash
# 查看 PyTorch 版本
python -c "import torch; print(torch.__version__)"
# 查看 CUDA 版本
python -c "import torch; print(torch.version.cuda)"
# 安装
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
# example 
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cpu+None.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cpu+None.html
pip install torch-geometric
```

example

```python
# 组建数据
# 边索引
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
# 节点特征
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
# 构建图
data = Data(x=x, edge_index=edge_index.t().contiguous())
# 查看节点
print(data.num_nodes)
# 参看节点特征维度
print(data.num_node_features)
# 可视化
g = to_networkx(data)
nx.draw(g, with_labels=g.nodes)
```

```python
class GATConv(in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True, edge_dim: Optional[int] = None, fill_value: Union[float, Tensor, str] = 'mean', bias: bool = True, **kwargs)
```

参数：

* in_channels： 输入节点特征维度；
* out_channels：输出节点的特征维度；
* heads：多头注意力；
* concat：`True`:多头注意力采用拼接的方式；`False`:多头注意力采用平均的方式；
* negative_slope：LeakyReLU的负斜率；
* dropout：注意力系数丢失概率；
* edge_dim：边缘特征；





















