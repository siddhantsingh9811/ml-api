	҉S??@҉S??@!҉S??@	?]???????]??????!?]??????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL҉S??@??z?"^C@1?s??m~@A\???4??I????ѥj@Y"?? >???rEagerKernelExecute 0*	?K7?Al@2U
Iterator::Model::ParallelMapV2y?JxB???!??????4@)y?JxB???1??????4@:Preprocessing2F
Iterator::Model??"1A??!?:?!D@)?????k??15a?
Z{3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?i? ?Ӫ?!?ퟦ%P7@);??u?+??1Jg?C3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJ????!T?
?,?@)V?F???1?w?h?/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????!???!ӎb?p/@)????!???1ӎb?p/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?z0)>>??!????M@)f?(?7??1y??x?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???r???!??‚0@)???r???1??‚0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap㥛? ???!??@\=@@)
??O?mi?1La?b???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?28.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?]??????I??n@A@Q??A??vP@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??z?"^C@??z?"^C@!??z?"^C@      ??!       "	?s??m~@?s??m~@!?s??m~@*      ??!       2	\???4??\???4??!\???4??:	????ѥj@????ѥj@!????ѥj@B      ??!       J	"?? >???"?? >???!"?? >???R      ??!       Z	"?? >???"?? >???!"?? >???b      ??!       JGPUY?]??????b q??n@A@y??A??vP@