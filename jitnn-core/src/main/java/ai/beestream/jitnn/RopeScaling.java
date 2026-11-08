/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn;

/** A rescaling of the rotary position embedding's frequencies, as Hugging Face's {@code rope_scaling}. */
public sealed interface RopeScaling {

    /** The frequency used in place of {@code frequency} (radians per position). */
    double apply(double frequency);

    /** Positions divided by {@code factor} (rope_type linear). */
    static RopeScaling linear(double factor) {
        return new Linear(factor);
    }

    /**
     * Llama 3.1's rescaling (rope_type llama3): frequencies with wavelengths above
     * originalContext / lowFreqFactor are divided by {@code factor}, those below
     * originalContext / highFreqFactor are kept, and the band between is interpolated.
     */
    static RopeScaling llama3(double factor, double lowFreqFactor, double highFreqFactor, int originalContext) {
        return new Llama3(factor, lowFreqFactor, highFreqFactor, originalContext);
    }

    /** Positions divided by {@code factor}. */
    record Linear(double factor) implements RopeScaling {
        @Override
        public double apply(double frequency) {
            return frequency / factor;
        }
    }

    /** Llama 3.1's rescaling; see {@link #llama3}. */
    record Llama3(double factor, double lowFreqFactor, double highFreqFactor, int originalContext) implements RopeScaling {
        @Override
        public double apply(double frequency) {
            double wavelength = 2 * Math.PI / frequency;
            if (wavelength < originalContext / highFreqFactor) {
                return frequency;
            }
            if (wavelength > originalContext / lowFreqFactor) {
                return frequency / factor;
            }
            double smooth = (originalContext / wavelength - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
            return (1 - smooth) * frequency / factor + smooth * frequency;
        }
    }
}
