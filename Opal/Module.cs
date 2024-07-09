using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Opal
{
	public interface IModule
	{
		public string Name { get; set; }
		public string[] Activators { get; set; }
		public string Process(string input, Context ctx);
	}
}
