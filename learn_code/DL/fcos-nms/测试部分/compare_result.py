import numpy as np


def stringify(l):
    return '_'.join([str(x) for x in l])

def compare(gt_path, ascend_out_path, dtype = np.float32):

    gt_data = np.fromfile(gt_path, dtype)
    ascend_out_data = np.fromfile(ascend_out_path, dtype)

    diff = np.abs(ascend_out_data - gt_data)
    max_diff_idx = np.argmax(diff)
    eps = 1e-2
    
    error_count = np.sum(diff>eps)
    error_rate = error_count / (gt_data.shape[0])
    
    if error_count > 0:
        print("[Compare Failed]: error rate: {:.2f}% = {}/{}".format(error_rate*100,error_count,gt_data.shape[0]))
    else:
        print("Compare Success!")

    print("Max diff: ground truth:{}, ascend output:{}".format(gt_data[max_diff_idx],ascend_out_data[max_diff_idx]))


def parse_reduce_sum_square(input_shape, output_shape, op_name="reduce_sum_square",
                     input_dtype=np.float32, output_dtype=np.float32):
    expect_path = f"./ground_truth/{op_name}_gt_{stringify(output_shape)}.bin"
    input_path = f"./input_data/{op_name}_in_{stringify(input_shape)}.bin"
    input_data = np.fromfile(input_path, dtype=input_dtype).reshape(input_shape)
    expect_data = np.fromfile(expect_path, dtype=output_dtype).reshape(output_shape)
    print(input_data.shape, expect_data.shape)
    try:
        output_path = f"./ascend_out/out.bin"
        output_data = np.fromfile(output_path, dtype=output_dtype).reshape(output_shape)
        print(f"in: {input_data.shape} gt: {expect_data.shape} out: {output_data.shape}")
        compare(expect_path, output_path)
    except:
        print("no output")
    finally:
        print("done")

if __name__ == "__main__":
    parse_reduce_sum_square([3, 4, 3], [3, 4, 1], "reduce_sum_squareMDC", np.float32, np.float32)
