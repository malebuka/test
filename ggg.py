import os
import gc
import torch
import tempfile
from collections import Counter, defaultdict
from safetensors.torch import save_file


# Если merge еще НЕ делал:
# model = model.merge_and_unload()

model.eval()
model.to("cpu")

gc.collect()
torch.cuda.empty_cache()

sd = model.state_dict()

print("TOTAL TENSORS:", len(sd))

print("\n=== DTYPES ===")
print(Counter(str(v.dtype) for v in sd.values() if isinstance(v, torch.Tensor)))

print("\n=== DEVICES ===")
print(Counter(str(v.device) for v in sd.values() if isinstance(v, torch.Tensor)))

print("\n=== LAYOUTS ===")
print(Counter(str(v.layout) for v in sd.values() if isinstance(v, torch.Tensor)))


print("\n=== BASIC TENSOR CHECK ===")
bad_basic = []

for k, v in sd.items():
    if not isinstance(v, torch.Tensor):
        bad_basic.append((k, "NOT_TENSOR", type(v)))
        continue

    try:
        info = {
            "name": k,
            "shape": tuple(v.shape),
            "dtype": str(v.dtype),
            "device": str(v.device),
            "layout": str(v.layout),
            "contiguous": v.is_contiguous(),
            "storage_offset": v.storage_offset(),
            "stride": v.stride(),
            "numel": v.numel(),
        }

        if v.device.type == "meta":
            bad_basic.append((k, "META_TENSOR", info))

        if v.layout != torch.strided:
            bad_basic.append((k, "NON_STRIDED_LAYOUT", info))

        if v.numel() == 0:
            bad_basic.append((k, "EMPTY_TENSOR", info))

    except Exception as e:
        bad_basic.append((k, "BASIC_CHECK_ERROR", repr(e)))

print("bad_basic count:", len(bad_basic))
for item in bad_basic[:50]:
    print(item)


print("\n=== SINGLE TENSOR SAFETENSORS SAVE CHECK ===")
bad_single = []

with tempfile.TemporaryDirectory() as tmp:
    for i, (k, v) in enumerate(sd.items()):
        if not isinstance(v, torch.Tensor):
            continue

        try:
            x = v.detach().cpu().contiguous()

            save_file(
                {k: x},
                os.path.join(tmp, "one.safetensors"),
                metadata={"format": "pt"},
            )

        except Exception as e:
            bad_single.append({
                "name": k,
                "shape": tuple(v.shape),
                "dtype": str(v.dtype),
                "device": str(v.device),
                "layout": str(v.layout),
                "contiguous": v.is_contiguous(),
                "storage_offset": v.storage_offset() if hasattr(v, "storage_offset") else None,
                "stride": v.stride() if hasattr(v, "stride") else None,
                "error": repr(e),
            })

            print("\nBAD SINGLE TENSOR:")
            print(bad_single[-1])
            break

print("bad_single count:", len(bad_single))


print("\n=== SHARED STORAGE CHECK ===")
storage_map = defaultdict(list)

for k, v in sd.items():
    if not isinstance(v, torch.Tensor):
        continue

    try:
        x = v.detach().cpu()
        storage = x.untyped_storage()

        storage_key = (
            storage.data_ptr(),
            storage.nbytes(),
            str(x.dtype),
        )

        storage_map[storage_key].append({
            "name": k,
            "shape": tuple(x.shape),
            "offset": x.storage_offset(),
            "stride": x.stride(),
            "contiguous": x.is_contiguous(),
        })

    except Exception as e:
        print("storage check error:", k, repr(e))

shared = {key: vals for key, vals in storage_map.items() if len(vals) > 1}

print("shared storage groups:", len(shared))

for idx, vals in enumerate(shared.values()):
    print(f"\nSHARED GROUP {idx + 1}:")
    for item in vals:
        print(item)
    if idx >= 20:
        break


print("\n=== FULL CLEAN STATE_DICT SAVE CHECK ===")
clean_sd = {}

for k, v in sd.items():
    if not isinstance(v, torch.Tensor):
        continue

    if v.device.type == "meta":
        print("SKIP META:", k)
        continue

    if v.layout != torch.strided:
        print("SKIP NON-STRIDED:", k, v.layout)
        continue

    clean_sd[k] = v.detach().cpu().contiguous()

try:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "full_clean.safetensors")
        save_file(clean_sd, path, metadata={"format": "pt"})
        print("FULL CLEAN SAVE OK")
except Exception as e:
    print("FULL CLEAN SAVE FAILED:")
    print(repr(e))


print("\n=== CHUNK / BINARY SEARCH SAVE CHECK ===")

items = list(clean_sd.items())

def can_save_range(start, end):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "chunk.safetensors")
            save_file(dict(items[start:end]), path, metadata={"format": "pt"})
        return True, None
    except Exception as e:
        return False, repr(e)

# Сначала ищем плохие крупные чанки
chunk_size = 50
bad_chunks = []

for start in range(0, len(items), chunk_size):
    end = min(start + chunk_size, len(items))
    ok, err = can_save_range(start, end)

    if not ok:
        bad_chunks.append((start, end, err))
        print(f"BAD CHUNK: {start}:{end}", err)

print("bad_chunks count:", len(bad_chunks))

# Потом бинарным поиском сужаем каждый плохой чанк
for start, end, err in bad_chunks:
    print(f"\n--- BINARY SEARCH IN CHUNK {start}:{end} ---")

    lo, hi = start, end

    while hi - lo > 1:
        mid = (lo + hi) // 2

        ok_left, err_left = can_save_range(lo, mid)
        ok_right, err_right = can_save_range(mid, hi)

        if not ok_left:
            print(f"bad left: {lo}:{mid}", err_left)
            hi = mid
        elif not ok_right:
            print(f"bad right: {mid}:{hi}", err_right)
            lo = mid
        else:
            print("Both halves OK separately. Problem is probably shared storage / interaction inside this chunk.")
            print("Chunk keys:")
            for j in range(start, end):
                name, tensor = items[j]
                print(j, name, tuple(tensor.shape), tensor.dtype, tensor.storage_offset(), tensor.stride())
            break

    if hi - lo == 1:
        name, tensor = items[lo]
        print("\nLIKELY BAD TENSOR:")
        print("index:", lo)
        print("name:", name)
        print("shape:", tuple(tensor.shape))
        print("dtype:", tensor.dtype)
        print("device:", tensor.device)
        print("layout:", tensor.layout)
        print("contiguous:", tensor.is_contiguous())
        print("storage_offset:", tensor.storage_offset())
        print("stride:", tensor.stride())
