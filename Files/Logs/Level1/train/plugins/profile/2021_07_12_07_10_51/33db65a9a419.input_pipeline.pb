	|E7?@|E7?@!|E7?@	_??2l??_??2l??!_??2l??"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL|E7?@?K?[??0@1{????|@Ar?	?OƼ?I??????l@Yj?????rEagerKernelExecute 0*	䥛? ?h@2F
Iterator::Model?,_?????!~qgQI@)?\QJV??1?=@?=@:Preprocessing2U
Iterator::Model::ParallelMapV2??l??)??!7?ߎ??4@)??l??)??17?ߎ??4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat4??@????!???:*?7@)??-???1ꖐ??}3@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU?#?????!??O???)@)U?#?????1??O???)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2??8*7??!??????H@)??J&???1'g1?q@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate_@/ܹ0??!?W???2@)?Fu:????1??U??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Qـ?!??{4??@)????Qـ?1??{4??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap34????!??jŰ4@)p?'v?u?1?貊?W@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?32.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9_??2l??I?{?UA@Q?>?5)PP@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?K?[??0@?K?[??0@!?K?[??0@      ??!       "	{????|@{????|@!{????|@*      ??!       2	r?	?OƼ?r?	?OƼ?!r?	?OƼ?:	??????l@??????l@!??????l@B      ??!       J	j?????j?????!j?????R      ??!       Z	j?????j?????!j?????b      ??!       JGPUY_??2l??b q?{?UA@y?>?5)PP@