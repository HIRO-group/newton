# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from newton.examples.hiro.example_poc import Example as example_base
import newton
import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True


class BaseExample:
    repeat = 10

    def setup(self, experiment):
        self.num_frames = 1000
        self.example = example_base(experiment=experiment,
                                    viewer=newton.viewer.ViewerNull(num_frames=self.num_frames))

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class GT(BaseExample):
    def setup(self):
        experiment = "gt"
        super().setup(experiment)

class B1(BaseExample):
    def setup(self):
        experiment = "b1"
        super().setup(experiment)

class B2(BaseExample):
    def setup(self):
        experiment = "b2"
        super().setup(experiment)

class B3(BaseExample):
    def setup(self):
        experiment = "b3"
        super().setup(experiment)

class OURS(BaseExample):
    def setup(self):
        experiment = "ours"
        super().setup(experiment)


if __name__ == "__main__":
    import argparse
    from newton.utils import run_benchmark

    benchmark_list = {
        "gt": GT,
        "b1": B1,
        "b2": B2,
        "b3": B3,

    }

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
