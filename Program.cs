using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/*
 * Adapted from Joe Albahari's "Writing a Neural Net from Scratch"  https://www.youtube.com/watch?v=z8DY5DndmxI
 * See also https://www.linqpad.net/
 * Joe's twitter https://twitter.com/linqpad?lang=en
 */

namespace IC
{

    static class Program
    {

        static void Main(string[] args)
        {
            Console.WriteLine("Starting");

            // LINQPad optimize+
            Sample[] trainingData, testingData;

            const int ImageWidthHeight = 28;

            trainingData = ImageSample.LoadTrainingImages();   // 50,000 training images
                testingData = ImageSample.LoadTestingImages();     // 10,000 testing images

                var net = new NeuralNet(ImageWidthHeight * ImageWidthHeight, 20, 10);

                var trainer = new Trainer(net).Dump();
                trainer.Train(trainingData, testingData, learningRate: .01, epochs: 10);

                //show failures
                var failures =
                    from testInfo in ImageSample.GetImageTestInfo(new FiringNet(net), testingData)
                    where !testInfo.IsCorrect
                    select new { testInfo.Image, testInfo.ImageSample.Label, testInfo.TotalLoss, testInfo.OutputValues };
                //worst 10
                //failures.OrderByDescending(f => f.TotalLoss).Take(10).Dump("Failures with highest loss");

            Console.ReadKey();

        }

    }
}

