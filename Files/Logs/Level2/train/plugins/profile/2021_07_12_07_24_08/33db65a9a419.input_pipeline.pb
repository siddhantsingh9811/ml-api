	7?Nx???@7?Nx???@!7?Nx???@	Z?(i"??Z?(i"??!Z?(i"??"?
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
lA@      ??!       "	ݚt[??}@ݚt[??}@!ݚt[??}@*      ??!       2	W?sD?K??W?sD?K??!W?sD?K??:	?\ik@?\ik@!?\ik@B      ??!       J	vQ???`??vQ???`??!vQ???`??R      ??!       Z	vQ???`??vQ???`??!vQ???`??b      ??!       JGPUYY?(i"??b qx??z?OA@y??@(tSP@