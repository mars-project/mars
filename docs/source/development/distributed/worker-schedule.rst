Execution in Worker
===================
A Mars worker consists of multiple processes to reduce the impact of the
notorious global interpreter lock (GIL) in Python. Executions run in separate
processes. To reduce unnecessary memory copy and inter-process communication,
shared memory is used to store computation results.

When an operand is being executed in a worker, it will first allocate memory.
Then data from other workers or from files already spilled to disk are loaded.
After that all data required are in memory and calculation can start. When
calculation is done, the worker then put the result into shared memory cache.
These four states can be seen in the graph below.

.. figure:: ../../images/worker-states.svg

Execution Control
-----------------
A Mars worker starts an ExecutionActor to control **all** the operands running
on the worker. It does not actually do calculation or data transfer itself, but
submit these actions to other actors.

OperandActors in schedulers submit an operand into workers through
``execute_graph`` calls. Then a callback is registered via
``add_finish_callback``. This design allows finish message be sent to different
places, which is necessary for failover.

ExecutionActor uses ``mars.promise`` module to handle multiple operands
simultaneously. Execution steps are chained via ``then`` method of the
``Promise`` class. When the final result is successfully stored, all registered
callbacks will be invoked. When exception raises in any chained promise, the
final exception handler registered with ``catch`` will try handling this
exception.

Operand Ordering
----------------
All operands in ``READY`` state are submitted into workers selected by the
scheduler. Therefore, the number of operands submitted to the worker is beyond
the capacity of the worker most of the time during execution, and the worker
need to sort these operands in order before picking some of the operands for
execution. This is done in TaskQueueActor in worker, where a priority queue is
maintained to store information of the operands. An allocator runs
periodically, trying to allocate resources for the operand at the head of the
queue till no free space left. The allocator is also triggered when a new
operand arrives, or an operand finishes execution.

Memory Management
-----------------
Mars worker manages two different parts of memory. The first is private memory
in every worker process, handled by every worker process. The second is shared
memory between all worker processes, handled by `plasma_store in Apache Arrow
<https://arrow.apache.org/docs/python/plasma.html>`_.

To avoid out-of-memory error in process memory, we introduce a worker-level
QuotaActor to allocate process memory. Before an operand starts execution, it
sends a memory batch request to the QuotaActor, asking for memory blocks for
its input and output chunks.  When memory quota left can satisfy the request,
the QuotaActor accepts the request. Otherwise the request is queued. After the
memory block is released, the allocation is freed and QuotaActor can accept
other requests.

Shared memory is handled by plasma\_store, which often takes of up to 50\% of
total memory. As there is no risk of out-of-memory, this part of memory is
allocated directly without quota requests. When shared memory is exhausted,
Mars worker tries to spill unused chunks into disk.

Chunks spill into disks may be used by later operands, and loading data from
disk into shared memory can be costly in IO, especially when the shared memory
is exhausted, and we need to spill other chunks to hold the loaded chunk.
Therefore, when data sharing is not needed, for instance, the input chunk is
only used by a single operand, it can be loaded into private memory instead of
shared memory. This can significantly reduce execution time.
