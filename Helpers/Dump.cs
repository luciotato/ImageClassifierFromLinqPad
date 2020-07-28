using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Newtonsoft.Json;

namespace IC {
    public static class ExtensionMethods {
        /// <summary>
        /// This is a simple (and lazy, read: effective) solution. Simply send your
        /// object to Newtonsoft serialize method, with the indented formatting, and
        /// you have your own Dump() extension method.
        /// </summary>
        /// <typeparam name="T">The object Type</typeparam>
        /// <param name="anObject">The object to dump</param>
        /// <param name="aTitle">Optional, will print this before the dump.</param>
        /// <returns>The object as you passed it</returns>
        public static T Dump<T>(this T anObject, string aTitle = "") {
            try {
                var pretty_json = JsonConvert.SerializeObject(anObject, Formatting.Indented);
                if (aTitle != "")
                    Console.Out.WriteLine(aTitle + ": ");

                Console.Out.WriteLine(pretty_json);
            }
            catch
            {
                Console.Out.WriteLine(anObject.GetType().FullName);
            }
            return anObject;
        }
    }
}