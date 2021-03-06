?	7?Nx???@7?Nx???@!7?Nx???@	Z?(i"??Z?(i"??!Z?(i"??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL7?Nx???@??>9
lA@1ݚt[??}@AW?sD?K??I?\ik@YvQ???`??rEagerKernelExecute 0*V-g@)       =2F
Iterator::Model???"R??!>?z?N?L@)????9??1?????=@:Preprocessing2U
Iterator::Model::ParallelMapV2TpxADj??!?T=???;@)TpxADj??1?T=???;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???A^??!?'N???0@)?k_@/ܙ?1w?AgRb+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate* ??q??!????4@)??6S!??1X_?:?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice7??????!{?? *@)7??????1{?? *@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?`6????!?Q?>?E@)?/?^|?~?1??d6Q@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??\QJv?!?kU?T@)??\QJv?1?kU?T@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapʨ2??A??!???Us5@)??QF\ Z?1LQC1????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?29.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Y?(i"??Ix??z?OA@Q??@(tSP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??>9
lA@??>9
lA@!??>9
lA@      ??!       "	ݚt[??}@ݚt[??}@!ݚt[??}@*      ??!       2	W?sD?K??W?sD?K??!W?sD?K??:	?\ik@?\ik@!?\ik@B      ??!       J	vQ???`??vQ???`??!vQ???`??R      ??!       Z	vQ???`??vQ???`??!vQ???`??b      ??!       JGPUYY?(i"??b qx??z?OA@y??@(tSP@?"H
-sequential_1/embedding_1/embedding_lookup/_17_Sendt?????!t?????"^
Cgradient_tape/sequential_1/embedding_1/embedding_lookup/Reshape/_27_Send?7??Hd??!,x>]*8??"k
?gradient_tape/sequential_1/conv1d_1/conv1d/Conv2DBackpropFilterConv2DBackpropFilter??}??=??!??>]?k??0"8
sequential_1/conv1d_1/conv1dConv2D4\#Ĺ??!~B??Ă??"i
>gradient_tape/sequential_1/conv1d_1/conv1d/Conv2DBackpropInputConv2DBackpropInput?U?P???!N??????0"=
$sequential_1/dropout_3/dropout/Mul_1MulR??RN??!???_??"K
2gradient_tape/sequential_1/dropout_3/dropout/Mul_1Mul)???խ??!???1?,??"J
,gradient_tape/sequential_1/conv1d_1/ReluGradReluGrad????<$??!???????"?
fgradient_tape/sequential_1/conv1d_1/conv1d/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer	Transpose????vW??!SR??hx??"=
#sequential_1/dropout_3/dropout/CastCast?
~?C??!??y????Q      Y@Y????=A@a??{aP@qSE?h???y{????G?"?

both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?29.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 