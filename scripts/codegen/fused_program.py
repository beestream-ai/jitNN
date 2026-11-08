# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Writes FusedKernels.fusedProgram1 .. fusedProgram6 (jitnn-core kernels/FusedKernels.java).

    python scripts/codegen/fused_program.py > /tmp/fused_program.java

prints the kernels; paste them over the generated block of FusedKernels.java. One kernel per
instruction count, so a chain of n element-wise instructions compiles only n instruction slots.
"""


def kernel(n):
    params = ", ".join(f"int p{j}" for j in range(n))
    body = []
    for j in range(n):
        args = "x0, x1, x2, x3, " + ", ".join(f"t{k}" if k < j else "0.0f" for k in range(6))
        a, b = f"params.get({2 * j})", f"params.get({2 * j + 1})"
        body.append(f"""            float t{j};
            {{
                float x = reg((p{j} >> 10) & 15, {args});
                float y = reg((p{j} >> 14) & 15, {args});
                int op = (p{j} >> 1) & 511;
                if ((p{j} & 1) == UNARY) {{
                    t{j} = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, {a}, {b}) : unaryLow(op, x, {a}, {b});
                }} else {{
                    t{j} = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, {a}, {b}) : binaryLow(op, x, y, {a}, {b});
                }}
            }}""")
    body += [f"            float t{j} = 0.0f;" for j in range(n, 6)]
    plural = "s" if n > 1 else ""
    return f"""    /** fusedFlat with the program in scalars (see above), for a chain of {n} instruction{plural}. */
    public static void fusedProgram{n}(KernelContext kc, FloatArray i0, FloatArray i1, FloatArray i2, FloatArray i3, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3,
            FloatArray params, int n, int inputs, {params}, int outs) {{
        int i = kc.globalIdx;
        if (i < n) {{
            float x0 = i0.get(i);
            float x1 = inputs > 1 ? i1.get(i) : 0.0f;
            float x2 = inputs > 2 ? i2.get(i) : 0.0f;
            float x3 = inputs > 3 ? i3.get(i) : 0.0f;
{chr(10).join(body)}
            store(i, outs, out0, out1, out2, out3, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5);
        }}
    }}
"""


if __name__ == "__main__":
    print("\n".join(kernel(n) for n in range(1, 7)))
