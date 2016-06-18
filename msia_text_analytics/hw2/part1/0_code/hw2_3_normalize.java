/** 
 * Author: Luis Steven Lin
 * 
 * Title: hw2_3_normalize
 * 
 * Purpose: Normalize using Stanford's CoreNLP lemmatizer
 *          
 * General Design:  Main method calls functions to read files, lemmatizes, print and save
 *                  a loop to get input and process inputs from user until it quits. 
 *
 *                  Input: classbios
 *                  Output: classbios_normalized.txt
 */

import java.io.*;
import java.util.*;

import org.apache.lucene.analysis.PorterStemFilter;
import org.apache.lucene.analysis.SimpleAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.en.KStemFilter;
import org.apache.lucene.util.Version;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.util.*;

/* References:
http://stackoverflow.com/questions/16516013/suggestions-for-a-word-lemmatizer
http://nlp.stanford.edu/software/corenlp.shtml
https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/ling/tokensregex/demo/TokensRegexRetokenizeDemo.java
http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
*/

public class hw2_3_normalize {
	
	
	public static void main(String[] args) throws IOException {
		
		// Inputs and Outputs
		String inputFile = "classbios.txt";
		String[] parts = inputFile.split("\\.");
		String outputFile = parts[0] + "_" + "normalized" + "." + parts[1];
		
	    // read some text in the text variable
	    //String text = "hi my name is U.S. USA";
	    
		// Read the file as list of strings
		ArrayList<String> lines_list = Functions.readFileLine(inputFile);
		//System.out.println(text);
		
		ArrayList<String> lemmas = new ArrayList<String>();
		
		// normalize each line
		for (String line:lines_list){
			
			if (!line.isEmpty()){
				
			    // Corenlp
				ArrayList<String> lemmas_corenlp = FunctionsCorenlp.lemmatize(line);
				String line_lemmas  = StringUtils.join(lemmas_corenlp ," ");
				lemmas.add(line_lemmas);
				
				
			} // end if
				
		} // end for 
		
		Functions.writeFile(outputFile, lemmas);
	       
	}// end main
		
} //end class