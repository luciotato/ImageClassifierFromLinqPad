using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IC {

    /// <summary>
    /// un Sample es lo que se feedea en la Net para obtener un resultado
    /// </summary>
    internal abstract class Sample {
        public double[] Data;
        public double[] ExpectedOutput;
        //public Func<double[], bool> IsOutputCorrect;
        public abstract bool IsOutputCorrect(double[] values);
    }

}
