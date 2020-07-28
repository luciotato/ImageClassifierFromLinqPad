using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace IC
{
    internal class ImageSample : Sample
    {
        const int categoryCount = 10;

        public byte Label;
        public byte[] Pixels;

        //constructor
        public ImageSample(byte label, byte[] pixels, int categoryCount)
        {
            Label = label;
            Pixels = pixels;
            Data = Util.PixelsToDouble(pixels);
            ExpectedOutput = Util.LabelToDoubleArray(label, categoryCount);
            //IsOutputCorrect = input => HelperMethods.IndexOfMax(input) == Label;
        }
        public override bool IsOutputCorrect(double[] values) {
            return Util.IndexOfMax(values) == Label;
        }


        public static ImageSample[] LoadTrainingImages() =>
            LoadImgeSampleArray(GetDataFilePath("Training Images", trainingImagesUri),
                GetDataFilePath("Training Labels", trainingLabelsUri),
                categoryCount);

        public static ImageSample[] LoadTestingImages() =>
            LoadImgeSampleArray(GetDataFilePath("Testing Images", testingImagesUri),
                GetDataFilePath("Testing Labels", testingLabelsUri),
                categoryCount);

        public static ImageSample[] LoadImgeSampleArray(string imgPath, string labelPath, int categoryCount)
        {
            $"Loading {System.IO.Path.GetFileName(imgPath)}...".Dump();
            var imgData = File.ReadAllBytes(imgPath);
            var header = imgData.Take(16).Reverse().ToArray();
            int imgCount = BitConverter.ToInt32(header, 8);
            int rows = BitConverter.ToInt32(header, 4);
            int cols = BitConverter.ToInt32(header, 0);

            return File.ReadAllBytes(labelPath)
                .Skip(8)  // skip header
                .Select((label, i) => new ImageSample(label, SliceArray(imgData, rows * cols * i + header.Length, rows * cols), categoryCount))
                .ToArray();
        }

        static byte[] SliceArray(byte[] source, int offset, int length)
        {
            var target = new byte[length];
            Array.Copy(source, offset, target, 0, length);
            return target;
        }

        static string SavedDataPath => System.IO.Path.Combine(Util.basePath, "saved.bin");

        const string
            trainingImagesUri = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            trainingLabelsUri = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            testingImagesUri = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            testingLabelsUri = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";

        static string GetDataFilePath(string filename, string uri)
        {
            if (!Directory.Exists(Util.basePath)) Directory.CreateDirectory(Util.basePath);
            string fullPath = System.IO.Path.Combine(Util.basePath, filename);

            if (!File.Exists(fullPath))
            {
                Console.Write($"Downloading {filename}... ");

                var buffer = new byte[0x10000];
                using (var ms = new MemoryStream(new WebClient().DownloadData(uri)))
                using (var inStream = new GZipStream(ms, CompressionMode.Decompress))
                using (var outStream = File.Create(fullPath))
                    while (true)
                    {
                        int len = inStream.Read(buffer, 0, buffer.Length);
                        if (len == 0) break;
                        outStream.Write(buffer, 0, len);
                    }

                Console.WriteLine("Done");
            }
            return fullPath;
        }

        internal static IEnumerable<TestInfo> GetImageTestInfo(FiringNet firingNet, Sample[] samples)
        {
            foreach (ImageSample sample in samples)
            {
                firingNet.FeedForward(sample.Data);
                yield return new TestInfo(sample, firingNet.OutputValues.ToArray());
            }
        }

    }
}
