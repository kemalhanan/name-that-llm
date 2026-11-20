"""Cut a GLM5-Next GGUF into layer-range segments, in the same layout as the public Kaggle segments
(jinghengwu/ntl-flash-iq3xxs-segs): every segment keeps all metadata, holds only its layers' tensors
and adds ntl.layer_start / ntl.layer_end / ntl.segment_index / ntl.segment_count.
token_embd goes in segment 0, output_norm and output in the last one. NextN (MTP) layers are dropped.

Only used to build the test fixture for ntl-segrun (the real segments are already public).

    python -I src/segrun/slice_gguf.py MODEL.gguf OUT_DIR 0,3,6,8
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'llama.cpp', 'gguf-py'))
import gguf  # noqa: E402


def main(src, out_dir, bounds):
    bounds = [int(x) for x in bounds.split(',')]
    reader = gguf.GGUFReader(src)
    arch = reader.get_field('general.architecture').contents()
    n_layer = reader.get_field(f'{arch}.block_count').contents()
    nextn = reader.get_field(f'{arch}.nextn_predict_layers')
    n_trunk = n_layer - (nextn.contents() if nextn else 0)
    assert bounds[0] == 0 and bounds[-1] == n_trunk, (bounds, n_trunk)
    os.makedirs(out_dir, exist_ok=True)
    n_seg = len(bounds) - 1
    for k in range(n_seg):
        s, e = bounds[k], bounds[k + 1]
        path = os.path.join(out_dir, f'seg{k:02d}-of-{n_seg:02d}-L{s}-{e}.gguf')
        writer = gguf.GGUFWriter(path, arch)
        for field in reader.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
                continue
            vt = field.types[0]
            st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            if vt == gguf.GGUFValueType.ARRAY and len(field.data) == 0:
                print(f'  skipping empty array {field.name}')
                continue
            writer.add_key_value(field.name, field.contents(), vt, sub_type=st)
        writer.add_uint32('ntl.layer_start', s)
        writer.add_uint32('ntl.layer_end', e)
        writer.add_uint32('ntl.segment_index', k)
        writer.add_uint32('ntl.segment_count', n_seg)
        writer.add_string('ntl.source_file', os.path.basename(src))

        def keep(name):
            if name.startswith('blk.'):
                il = int(name.split('.')[1])
                return s <= il < e
            if name.startswith('token_embd'):
                return k == 0
            return k == n_seg - 1  # output_norm, output

        tensors = [t for t in reader.tensors if keep(t.name)]
        for t in tensors:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for t in tensors:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
        writer.close()
        print(f'{path}: layers [{s}, {e}), {len(tensors)} tensors')


if __name__ == '__main__':
    main(*sys.argv[1:4])
