import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;

/* References:
http://stackoverflow.com/questions/16027229/reading-from-a-text-file-and-storing-in-a-string
http://stackoverflow.com/questions/9213564/method-call-to-another-file
http://www.programcreek.com/2011/03/java-read-a-file-line-by-line-code-example/
*/
	
public class Functions {
	
	/**
	 * writeFile
	 * The function writes the information stored in the array to
	 * a txt file
	 * @param String fileName 
	 * @param ArrayList<String> tokens
	 * @throws IOException
	 */
	public static void writeFile(String fileName, ArrayList <String > tokens) throws IOException {
		
		PrintWriter out = new PrintWriter(fileName);
		
		for (int i=0; i < tokens.size(); i++){
			out.printf("%s%n",tokens.get(i));
		}
		
		out.close();	
		
	} // end writeFile
	
	/**
	 * writeFileMap
	 * The function writes the information stored in the map, one line per element in list
	 * for each key in the map. So output is: key1: element1 in list key1
	 * 										  key2: element1 in list key2
	 * 
	 * 										  key1: element2 in list key1
	 * 										  key2: element2 in list key2   etc
	 * a txt file
	 * @param String fileName 
	 * @param Map<String, ArrayList<String>> map
	 * @throws IOException
	 */
	public static void writeFileMap(String fileName, Map<String, ArrayList<String>> map) throws IOException {
		
		PrintWriter out = new PrintWriter(fileName);
		
		ArrayList<String> keys =  new ArrayList<String>(map.keySet());
		Collections.sort( keys );
		

		for (int i=0; i < map.get(keys.get(0)).size(); i++){
						
			for (String key : keys) {
				out.printf("%-18s %s %s %n",key,"==>", map.get(key).get(i));
			}

			out.printf("%n"); //newline
			
		}
		
		out.close();	
		
	} // end writeFile
	
	
	/**
	 * readFile
	 * Function reads file an return it as an entire string
	 * @param fileName <String>
	 * @return text <String>
	 * @throws IOException
	 */

	public static String readFile(String fileName) throws IOException {
	    BufferedReader br = new BufferedReader(new FileReader(fileName));
	    try {
	        StringBuilder sb = new StringBuilder();
	        String line = br.readLine();

	        while (line != null) {
	            sb.append(line);
	            sb.append("\n");
	            line = br.readLine();
	        }
	        return sb.toString();
	    } finally {
	        br.close();
	    }
	} // end readFile
	

	/**
	 * readFileLine
	 * Function reads file and returns each line as an element in a list
	 * @param fileName <String>
	 * @return sentences <ArrayList<String>>
	 * @throws IOException
	 */
	
	public static  ArrayList<String>  readFileLine(String fileName) throws IOException {
		
		ArrayList<String> sentences = new ArrayList<String>();
		
		//InputStreamReader is always a safer choice than FileReader
		FileInputStream fis = new FileInputStream(fileName);
	 
		//Construct BufferedReader from InputStreamReader
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
	 
		String line;
		while ((line = br.readLine()) != null) {
			sentences.add(line);
		}
	 
		br.close();
		
		return sentences;
	} // end function
	
	
	/**
	 * addBrackets
	 * Function adds [ element ] for every element in list and returns a list
	 * @param tokens <ArrayList<String>>
	 * @return tokens_bracket <ArrayList<String>>
	 * @throws IOException
	 */
	
	public static  ArrayList<String>  addBrackets(ArrayList<String> tokens) throws IOException {
		
		ArrayList<String> tokens_bracket = new ArrayList<String>();
		
		String token_bracket;
		for (String token:tokens ){
			
			token_bracket = String.format("[%s]", token);
			
			tokens_bracket.add(token_bracket);
				
		}
		

	
		
		return tokens_bracket;
	} // end function
	
	

} // end class
