using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.Formats.Bmp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.Shapes;

namespace Basic
{
    class Program
    {
        const float TwoPI = MathF.PI * 2;
        const int W = 128;
        const int H = 128;
        const int N = 128;
        const int MaxStep = 10;
        const float MaxDistance = 2f;
        const float Epsilon = 1e-6f;
        static Circle _circle = new Circle { Center = new Vector2(0.5f, 0.5f), Radius = 0.1f };

        static Random _random = new Random();
        static readonly float[] _rands = Enumerable.Range(0, N * H).Select(x => (float)_random.NextDouble()).ToArray();

        struct Circle
        {
            public Vector2 Center;
            public float Radius;
        }

        static float Trace(Vector2 point, Vector2 lightDir)
        {
            var m = point - _circle.Center;
            var c = Vector2.Dot(m, m) - MathF.Pow(_circle.Radius, 2);
            var b = Vector2.Dot(m, lightDir);
            var d = b * b - c;
            return d >= 0 ? 1.0f : 0;
        }

        static float Sample(Vector2 point, int seed)
        {
            float sum = 0.0f;
            for (int i = 0; i < N; i++)
            {
                float a = TwoPI * (i + _rands[(seed + i) % _rands.Length]) / N;
                Vector2 dir = new Vector2(MathF.Cos(a), MathF.Sin(a));
                sum += Trace(point, dir);
            }
            return sum / N;
        }

        static void Main(string[] args)
        {
            var encoder = new PngEncoder { ColorType = PngColorType.Grayscale };
            foreach (var i in Enumerable.Range(0, 60))
            {
                var r = i / 120f;

                _circle.Radius = r;
                using (var img = new Image<Gray8>(W, H))
                {
                    int seed = 0;
                    for (int y = 0; y < H; y++)
                    {
                        Parallel.For(0, W, x =>
                        {
                            ref var pixel = ref img.GetPixelRowSpan(y)[x];
                            var p = new Vector2((float)x / W, (float)y / H);

                            var value = (byte)MathF.Min(Sample(p, Interlocked.Increment(ref seed)) * 255.0f, 255.0f);
                            pixel = new Gray8(value);
                        });
                    }

                    using (var fs = File.OpenWrite($"data/o_{i}.png"))
                        img.SaveAsPng(fs, encoder);
                }
            }
        }
    }
}
