namespace Opal
{
	public class Bot
	{
		public string Name { get; set; } = "";

		public const string	VERSION = "0.0.0";

		private IModule[] modules = [];
		private Context	context = new();

		internal Bot(string name) 
		{
			Name = name;
		}

		public string Respond(string input)
		{
			string output = "";

			foreach (IModule module in modules)
			{
				

			}

			return output;
		}
	}
}
