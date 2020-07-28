using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IC
{
    static internal class Util
    {
        // Helper methods
        static internal readonly string basePath = System.IO.Path.Combine(
                    Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData),
                            "LINQPad Machine Learning", "MNIST digits");


        //static double[] ToDouble(byte[] data) => data.Select(p => ((double)p+20) / (255+20)).ToArray();
        //internal static double[] PixelsToDouble(byte[] data) => data.Select(p => ((double)p-127) / 128 ).ToArray();
        internal static double[] PixelsToDouble(byte[] data) => data.Select(p => (double)p / 255).ToArray();

        internal static double[] LabelToDoubleArray(byte label, int categoryCount)
            //=>
            //Enumerable.Range(0, categoryCount).Select(i => i == label ? 1d : 0).ToArray();
            {
            var result = new double[categoryCount];
            result[label] = 1d; //el resto queda en cero
            return result;
        }

        static internal int IndexOfMax(double[] values)
        {
            double max = 0;
            int indexOfMax = 0;
            for (int i = 0; i < values.Length; i++)
                if (values[i] > max)
                {
                    max = values[i];
                    indexOfMax = i;
                }
            return indexOfMax;
        }

    }

}

