using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IC
{
    internal class TestInfo
    {
        public readonly ImageSample ImageSample;
        public readonly double[] OutputValues;

        public bool IsCorrect => ImageSample.IsOutputCorrect(OutputValues);

        public double TotalLoss => OutputValues
            .Select((v, i) => (v - (i == ImageSample.Label ? 1 : 0)) * (v - (i == ImageSample.Label ? 1 : 0)) / 2)
            .Sum();

        Lazy<System.Drawing.Image> _image;
        public System.Drawing.Image Image {
            get {
                if (_image == null) return null;
                return _image.Value;
            }
        }

        public TestInfo(ImageSample imageSample, double[] outputValues)
        {
            ImageSample = imageSample;
            OutputValues = outputValues;
            //_image = new Lazy<System.Drawing.Image>(() => ToImage(ImageSample.Pixels, 0, ImageWidthHeight, ImageWidthHeight));
        }
    }

}
