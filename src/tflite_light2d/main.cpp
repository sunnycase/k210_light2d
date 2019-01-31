/* Copyright 2018 Canaan Inc.
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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <devices.h>
#include <filesystem.h>
#include <memory>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <FreeRTOS.h>
#include <task.h>

extern "C"
{
#include "lcd.h"
}

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include "incbin.h"
INCBIN(model, "../src/tflite_light2d/basic.tflite");

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        while (1)                                                \
            ;                                                    \
    }

extern const char *labels[];
namespace tflite {
    namespace ops {
        namespace builtin {
            TfLiteRegistration* Register_TRANSPOSE_CONV();
            TfLiteRegistration* Register_MUL();
            TfLiteRegistration* Register_ADD();
            TfLiteRegistration* Register_RESHAPE();
        }
    }
}

using namespace tflite::ops::builtin;

class NeededOpResolver : public MutableOpResolver
{
public:
    NeededOpResolver()
    {
        AddBuiltin(BuiltinOperator_TRANSPOSE_CONV, Register_TRANSPOSE_CONV());
        AddBuiltin(BuiltinOperator_MUL, Register_MUL());
        AddBuiltin(BuiltinOperator_ADD, Register_ADD());
        AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
    }

    const TfLiteRegistration* FindOp(tflite::BuiltinOperator op,
        int version) const override
    {
        return MutableOpResolver::FindOp(op, version);
    }

    const TfLiteRegistration* FindOp(const char* op, int version) const override
    {
        return MutableOpResolver::FindOp(op, version);
    }
};

uint16_t lcd_gram[128 * 128] __attribute__((aligned(128)));

void vTask1(void* arg)
{
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromBuffer((const char *)model_data, model_size);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    printf("model built\n");

    // Build the interpreter
    NeededOpResolver resolver;
    InterpreterBuilder builder(*model.get(), resolver);
    std::unique_ptr<Interpreter> interpreter;
    builder(&interpreter, 1);
    printf("interpreter built\n");
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    // Fill input buffers
    int input = interpreter->inputs()[0];

    int i = 0;
    int step = 1;

    while (1)
    {
        i = i + step;
        if (i >= 5 || i <= 0)
            step = -step;
        interpreter->typed_tensor<float>(input)[0] = i / 10.f;

        // Run inference
        TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

        auto output_it = interpreter->typed_output_tensor<float>(0);

        for (size_t i = 0; i < 128 * 128; i++)
        {
            auto gray = std::clamp((int)(output_it[i] * 255), 0, 255);
            auto rgb = ((gray & 0b11111000) << 8) | ((gray & 0b11111100) << 3) | (gray >> 3);
            size_t d_i = i % 2 ? (i - 1) : (i + 1);
            lcd_gram[d_i] = rgb;
        }

        lcd_draw_picture(100, 60, 128, 128, reinterpret_cast<uint32_t*>(lcd_gram));

        vTaskDelay(5);
    }
}

int main()
{
    lcd_init();
    printf("lcd init\n");
    
    lcd_clear(BLACK);

    xTaskCreate(vTask1, "vTask1", 1024 * 60, NULL, 3, NULL);
    while(1);
}
