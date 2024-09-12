import re

# 输入的文本
text = """
 * TopK=1 (0.672 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=24,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=2 (0.673 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=26,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=3 (0.68 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=28,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=4 (0.681 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=32,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=5 (0.686 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=30,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=6 (0.696 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=36,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=7 (0.7 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=22,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=8 (0.703 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=34,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=9 (0.711 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=40,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=10 (0.715 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=38,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=11 (0.727 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=44,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=12 (0.729 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=20,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=13 (0.734 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=42,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=14 (0.75 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=48,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=15 (0.751 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=46,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=16 (0.782 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=50,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=17 (0.801 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=54,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=18 (0.809 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=52,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=19 (0.812 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=56,n_dim_per_split=512),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
"""

# 正则表达式模式
pattern = r'\(([\d\.]+) ms\).*?gather_rs_ctas=(\d+)'

# 使用正则表达式提取特征
matches = re.findall(pattern, text)

# 将提取的特征转换为包含时间值和 gather_rs_ctas 的元组列表
features = [(float(ms), int(ctas)) for ms, ctas in matches]

# 根据 gather_rs_ctas 对列表进行排序
sorted_features = sorted(features, key=lambda x: x[1])

# 将结果写入到文本文件
with open('tmp_output.txt', 'w') as file:
    for ms, ctas in sorted_features:
        file.write(f"{ms}\n")

print("Data has been written to output.txt")