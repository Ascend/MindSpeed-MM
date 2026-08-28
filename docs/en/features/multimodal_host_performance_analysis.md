# Multimodal Host Performance Analysis

## 1. Background

With the rapid development of multimodal large models (such as Qwen3.5 and Wan2.2), model size and complexity continue to grow, leading to ever-increasing demand for computational resources. In multimodal training and inference, the host side (typically referring to the CPU and its memory system) plays a critical role: it is responsible for preprocessing operations such as loading, parsing, and augmentation of multimodal data; manages data such as model parameters, optimizer states, and intermediate activations; interacts with the storage system; and performs data transfer with acceleration devices (GPU/NPU).

As model size increases and multimodal data becomes increasingly complex, performance bottlenecks on the host side gradually become a key factor limiting overall system efficiency. Therefore, in-depth analysis and optimization of host-side performance are of great significance for improving the training and inference efficiency of multimodal large models.

## 2. Current Status and Challenges

### 2.1 Overview of Multimodal Data Processing

Multimodal data processing involves multiple modalities, each with unique processing requirements:

| Modality | Core Processing Steps                       | Main Overhead       |
| -- | ---------------------------- | ---------- |
| Text | Tokenization, length truncation/padding, feature encoding | Moderate CPU, low memory |
| Image | JPEG/PNG decoding, resize, normalization, data augmentation  | High CPU, moderate memory |
| Video | Video decoding, frame sampling, resize, temporal processing         | Very high CPU, high memory |
| Audio | Audio decoding, Mel-spectrogram extraction, temporal processing            | High CPU, moderate memory |

### 2.2 Performance Bottleneck Analysis

#### 2.2.1 CPU Bottlenecks

Multimodal data processing involves a large number of compute-intensive operations (such as image resizing and video decoding). Traditional serial processing fails to fully utilize multi-core CPUs, while multithread creation and synchronization also introduce additional overhead.

#### 2.2.2 Memory Bottlenecks

High-resolution images and long video sequences lead to excessive memory usage, and frequent memory allocation and deallocation cause fragmentation. In multimodal data processing, intensive memory access makes memory bandwidth a bottleneck.

#### 2.2.3 I/O Bottlenecks

The read speed of large-scale multimodal datasets limits overall performance. Mechanical hard drives or network storage introduce high access latency, and traditional serial I/O fails to leverage the parallel capabilities of the storage system.

#### 2.2.4 Communication Bottlenecks

The large size of multimodal data leads to significant transfer overhead between the host and the device, and transfer latency and synchronization waiting directly affect training efficiency.

### 2.3 Special Challenges of Multimodal Models

- **Modal heterogeneity**: Different modalities have significantly different data formats and processing methods, making unified processing difficult.
- **Data alignment**: Different modal data must be aligned in both time and space.
- **Unbalanced resource requirements**: Different modalities have significantly different computational and memory requirements, making resource allocation difficult.
- **High difficulty in dynamic batching**: Multimodal data exhibits large variations in length and complexity.

## 3. Methodology: A Framework for Analyzing Symptoms to Root Causes

### 3.1 Four-Step Performance Analysis

#### 3.1.1 Symptom Observation

Identify performance anomalies through monitoring and logging:

- **Training speed drop**: Fewer samples processed per second.
- **Abnormal CPU usage**: Usage too high or too low.
- **Abnormal memory usage**: Usage too high or memory leaks.
- **Long IO wait time**: Data loading takes too long.
- **High communication latency**: Host-to-device communication time accounts for a high proportion.

#### 3.1.2 Data Collection

Collect multidimensional performance data:

- **System metrics**: CPU usage, memory usage, I/O throughput, network bandwidth
- **Framework metrics**: Data loading time, preprocessing time, computation time, communication time
- **Application metrics**: Per-batch processing time, sample throughput

#### 3.1.3 Bottleneck Identification

Locate bottlenecks through three types of analysis:

- **Time analysis**: Analyze time proportions of each stage to find the most time-consuming stage.
- **Resource analysis**: Analyze resource usage to find resource bottlenecks.
- **Dependency analysis**: Analyze dependencies between components to identify the critical path.

#### 3.1.4 Cause Analysis

Dig into the root causes from four levels:

| Level | Common Cause | Analysis Method |
| -- | --------- | --------------------- |
| Algorithm | Low efficiency of data processing algorithms | Algorithm complexity analysis, benchmarking |
| Implementation | Insufficiently optimized code implementation | Code review, profiling |
| System | Unreasonable hardware resource configuration | Resource utilization analysis |
| Architecture | System architecture design flaws | Data flow analysis, architecture review |

### 3.2 Performance Analysis Tools

#### 3.2.1 System-Level Tools

- **CPU**: `top`, `htop`, `pidstat`
- **Memory**: `free`, `vmstat`, `pmap`
- **I/O**: `iostat`, `iotop`, `dstat`
- **Network**: `netstat`, `ss`, `iftop`

#### 3.2.2 Framework-level Tools

- **PyTorch**: `torch.profiler`, `torch.utils.bottleneck`
- **TensorFlow**: `tf.profiler`, TensorBoard

#### 3.2.3 Custom Profiler

Develop a dedicated analysis tool for the multimodal data processing pipeline to measure the time consumption and proportion of each stage:

```python
import time

class MultimodalProfiler:
    def __init__(self):
        self._times = {}
        self._starts = {}

    def start(self, name):
        self._starts[name] = time.time()

    def end(self, name):
        if name in self._starts:
            self._times[name] = time.time() - self._starts.pop(name)

    def report(self):
        print("=== Host Performance Report ===")
        total = sum(self._times.values())
        for name, t in sorted(self._times.items(), key=lambda x: -x[1]):
            pct = t / total * 100 if total > 0 else 0
            print(f"  {name:30s}: {t:8.4f}s ({pct:5.1f}%)")
        print(f"  {'Total':30s}: {total:8.4f}s")

# Usage example
profiler = MultimodalProfiler()
profiler.start("text_tokenization")
# ... text processing ...
profiler.end("text_tokenization")

profiler.start("image_decode_resize")
# ... image processing ...
profiler.end("image_decode_resize")

profiler.start("data_transfer_h2d")
# ... Data transfer ...
profiler.end("data_transfer_h2d")

profiler.report()
```

### 3.3 Case Studies

#### 3.3.1 Case 1: Sudden Training Speed Drop

**Symptom**: Model throughput dropped sharply from 200 samples/s to 50 samples/s.

**Analysis process**:

1. **Symptom  observation**: CPU usage dropped from 70% to 20%, and I/O wait time increased significantly.
2. **Data collection**: System metrics (CPU/IO/memory) and framework metrics (time spent in each stage) collected.
3. **Bottleneck identification**: The data loading time surged from 0.1s to 1.5s, becoming the main bottleneck.
4. **Cause analysis**: Checked data loading code and found serial I/O with no prefetching mechanism; as training progressed, the file pointer moved to non-contiguous disk regions, increasing random I/O.

**Solution**: Increase data loading threads → Implement data prefetching → Use mmap technology → Optimize the data storage format to reduce random I/O

#### 3.3.2 Case 2: Continuous Memory Growth

**Symptom**: Memory usage gradually increased from 50% to over 95%, eventually causing OOM.

**Analysis process**:

1. **Symptom observation**: Memory usage continued to grow monotonically.
2. **Data collection**: Memory usage trends and allocation/release records collected.
3. **Bottleneck identification**: Memory allocated during image processing was not released promptly.
4. **Cause analysis**: Image processing code used Python lists to store processed data but did not clear them after processing; the garbage collector did not reclaim memory in time.

**Solution**: Timely release unused memory → Use a memory pool to manage allocations → Optimize data structures to reduce memory usage → Periodically trigger `gc.collect()`.

## 4. Techniques and Code Implementation

### 4.1 Data Processing Optimization

#### 4.1.1 Parallel Data Processing

Leverage multi-core CPUs to process multimodal data in parallel and improve throughput:

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import cv2

def process_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    return np.array(image) / 255.0

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (224, 224)) / 255.0)
    cap.release()
    return frames

def process_batch_parallel(data_batch):
    with ThreadPoolExecutor(max_workers=4) as ex:
        images = list(ex.map(process_image, [d['image_path'] for d in data_batch]))
    with ThreadPoolExecutor(max_workers=4) as ex:
        videos = list(ex.map(process_video, [d['video_path'] for d in data_batch]))
    return [{'text': d['text'], 'image': img, 'video': vid}
            for d, img, vid in zip(data_batch, images, videos)]
```

#### 4.1.2 Data Prefetching

Read data asynchronously in advance to reduce I/O waiting:

```python
import threading
import queue

class PrefetchDataLoader:
    def __init__(self, dataset, batch_size, num_workers=4, prefetch_factor=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=prefetch_factor * num_workers)
        self._stop = threading.Event()
        self._workers = [threading.Thread(target=self._worker_fn, args=(i,), daemon=True)
                         for i in range(num_workers)]
        for w in self._workers:
            w.start()

    def _worker_fn(self, wid):
        step = len(self._workers)
        while not self._stop.is_set():
            for bid in range(wid, len(self.dataset), step):
                batch = [self.dataset[bid * self.batch_size + i]
                         for i in range(self.batch_size)
                         if bid * self.batch_size + i < len(self.dataset)]
                if batch:
                    self.queue.put(batch)

    def __iter__(self):
        for _ in range(len(self)):
            yield self.queue.get(timeout=10)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def shutdown(self):
        self._stop.set()
```

### 4.2 Memory Management Optimization

#### 4.2.1 Memory Pool

Preallocate memory buffers to reduce the overhead of frequent allocation/deallocation:

```python
import numpy as np

class MemoryPool:
    def __init__(self, dtype=np.float32, block_mb=100):
        self.dtype = dtype
        self.block_size = block_mb * 1024 * 1024 // np.dtype(dtype).itemsize
        self._blocks = []
        self._free = []

    def allocate(self, size):
        for i, blk in enumerate(self._free):
            if blk.size >= size:
                return self._free.pop(i)[:size]
        blk = np.empty(max(size, self.block_size), dtype=self.dtype)
        self._blocks.append(blk)
        return blk[:size]

    def free(self, arr):
        self._free.append(arr)

    def clear(self):
        self._blocks.clear()
        self._free.clear()
```

#### 4.2.2 Data Compression

Compress non-critical data for storage to save host memory:

```python
import zlib
import numpy as np

def compress_array(arr):
    return zlib.compress(arr.tobytes())

def decompress_array(data, dtype, shape):
    return np.frombuffer(zlib.decompress(data), dtype=dtype).reshape(shape)

# Usage example
arr = np.random.rand(1000, 1000).astype(np.float32)
compressed = compress_array(arr)
print(f"Compression ratio: {len(compressed) / arr.nbytes:.2%}")
restored = decompress_array(compressed, arr.dtype, arr.shape)
assert np.allclose(arr, restored)
```

### 4.3 I/O Optimization

#### 4.3.1 Parallel I/O

Read files in parallel using multiple threads to improve I/O throughput:

```python
import threading

def parallel_read(file_paths, num_threads=4):
    results, lock = {}, threading.Lock()

    def worker(files):
        for fp in files:
            try:
                with open(fp, 'rb') as f:
                    with lock:
                        results[fp] = f.read()
            except Exception as e:
                with lock:
                    results[fp] = None

    threads = []
    for i in range(min(num_threads, len(file_paths))):
        t = threading.Thread(target=worker, args=(file_paths[i::num_threads],))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results
```

#### 4.3.2 Memory Mapping (mmap)

Directly map large files into the virtual address space to avoid explicit copying:

```python
import mmap
import numpy as np

def load_with_mmap(file_path, dtype=np.float32):
    with open(file_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data = np.frombuffer(mm, dtype=dtype)
        return data, mm  # The caller must call mm.close() after use.
```

### 4.4 Communication Optimization

#### 4.4.1 Batch Transfer

Reduce the number of host-to-device communication operations to lower communication overhead:

```python
import torch
import numpy as np

def batch_to_device(data_list, device):
    if isinstance(data_list[0], np.ndarray):
        return torch.from_numpy(np.stack(data_list, axis=0)).to(device)
    elif isinstance(data_list[0], torch.Tensor):
        return torch.stack(data_list, dim=0).to(device)
    return [d.to(device) for d in data_list]
```

#### 4.4.2 Asynchronous Transfer

Use `non_blocking` transfer to overlap communication and computation, hiding communication latency:

```python
import torch

def async_to_device(data, device):
    if isinstance(data, np.ndarray):
        t = torch.empty(data.shape, dtype=torch.float32, device=device)
        t.copy_(torch.from_numpy(data), non_blocking=True)
        return t
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    return data

# Usage: Start asynchronous transfer, perform other computations on the device, and finally call cuda.synchronize().
```

## 5. Multimodal Model Host Performance Optimization Practices

### 5.1 Qwen3.5

The core optimization of the Qwen3.5 multimodal version lies in the parallelization of the data processing pipeline and memory management.

**Original flow**: Serial read of text/images → Serial processing → Serial transfer to device

**Optimized flow**: Parallel read → Parallel processing → Batched asynchronous transfer

```python
import torch
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

class Qwen35Dataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        with open(r['text'], 'r', encoding='utf-8') as f:
            text = f.read()
        return text, Image.open(r['image'])

def collate_qwen35(batch):
    texts, images = zip(*batch)
    with ThreadPoolExecutor(max_workers=4) as ex:
        texts = list(ex.map(lambda t: t[:512], texts))
    with ThreadPoolExecutor(max_workers=4) as ex:
        images = list(ex.map(lambda im: np.array(im.resize((224, 224))) / 255.0, images))
    return (
        torch.tensor([len(t) for t in texts]),
        torch.tensor(np.stack(images)).permute(0, 3, 1, 2)
    )

# Use DataLoader with pin_memory=True and non_blocking transfer.
loader = DataLoader(Qwen35Dataset(records), batch_size=32, num_workers=4,
                    collate_fn=collate_qwen35, pin_memory=True)
device = torch.device('cuda')
for lengths, images in loader:
    lengths, images = lengths.to(device, non_blocking=True), images.to(device, non_blocking=True)
    # output = model(lengths, images)
```

### 5.2 Wan2.2

The challenge of Wan2.2 lies in the host overhead of video decoding and multimodal fusion.

```python
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor

def decode_video(path, max_frames=32):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total // max_frames)
    frames, idx = [], 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(cv2.resize(frame, (224, 224)) / 255.0)
        idx += 1
    cap.release()
    while len(frames) < max_frames:
        frames.append(np.zeros((224, 224, 3), dtype=np.float32))
    return np.array(frames)

def batch_decode_videos(paths, max_frames=32, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda p: decode_video(p, max_frames), paths))
    return torch.tensor(np.stack(results)).permute(0, 1, 4, 2, 3)
```

The design of the multimodal fusion module should incorporate feature caching to avoid repeated projection computation:

```python
import torch.nn as nn

class CachedFusion(nn.Module):
    def __init__(self, text_dim, img_dim, vid_dim, out_dim):
        super().__init__()
        self.t_proj = nn.Linear(text_dim, out_dim)
        self.i_proj = nn.Linear(img_dim, out_dim)
        self.v_proj = nn.Linear(vid_dim, out_dim)
        self.weights = nn.Parameter(torch.ones(3))
        self._cache = [None, None, None]

    def forward(self, t_feat, i_feat, v_feat):
        if self._cache[0] is None or t_feat.shape[0] != self._cache[0].shape[0]:
            self._cache[0] = self.t_proj(t_feat)
            self._cache[1] = self.i_proj(i_feat)
            self._cache[2] = self.v_proj(v_feat)
        w = torch.softmax(self.weights, dim=0)
        return w[0] * self._cache[0] + w[1] * self._cache[1] + w[2] * self._cache[2]
```

## 6. Best Practices and Recommendations

### 6.1 Hardware Configuration

| Model Size    | CPU Cores | Memory         | Storage          | Load Threads  | Batch Size   |
| ------- | ------ | ---------- | ----------- | ----- | ----- |
| < 10B    | 8-16   | 64-128 GB  | NVMe SSD    | 4-8   | 32-64 |
| 10B-70B | 16-32  | 128-256 GB | NVMe SSD Array | 8-16  | 16-32 |
| > 70B    | 32+    | 256 GB+    | Distributed Storage       | 16-32 | 8-16  |

### 6.2 Software Configuration

- **Operating system**: Set `vm.swappiness=1` and `vm.dirty_ratio=10`; use the ext4/xfs file system.
- **PyTorch**: Enable `torch.backends.cudnn.benchmark` and use a DataLoader with `pin_memory=True`.
- **Data format**: Prefer columnar storage formats such as LMDB/TFRecord to reduce small-file I/O.

### 6.3 Tuning Process

1. **Baseline testing**: Establish a performance baseline and define optimization goals.
2. **Bottleneck identification**: Use the profilers and system tools mentioned above to locate bottlenecks.
3. **Implement optimization**: Optimize in the order of "data processing → memory → I/O → communication".
4. **Effect evaluation**: Compare metrics before and after optimization to quantify improvements.
5. **Continuous iteration**: Fine-tune based on runtime observations.

## 7. Conclusion and Outlook

Multimodal host performance analysis is a key aspect of optimizing large model training efficiency. Key conclusions are as follows:

- **Data processing is the main bottleneck**: Multimodal preprocessing consumes significant CPU time, and parallelization yields the highest gains.
- **Memory management cannot be overlooked**: Memory pools and compression can effectively reduce peak usage and prevent OOM.
- **I/O optimization has great potential**: mmap, prefetching, and parallel I/O can significantly reduce training wait time.
- **Communication optimization adds further gains**: Batched + asynchronous transfers overlap communication with computation.

Future directions: Automated performance analysis tools for self-identification of bottlenecks; intelligent resource scheduling that adapts configurations dynamically based on model characteristics; exploration of specialized hardware like NPUs/TPUs for accelerating multimodal data processing; and research on Host-side collaborative optimization strategies in distributed environments.
