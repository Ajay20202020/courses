import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.AttributeSource;
import org.apache.lucene.util.Version;

public class FunctionsLucene {
	
	/**
	 * normalize
	 * The function normalizes a stream, prints the tokens
	 * and returns an array list with the tokens
	 * @param stream <TokenStream>
	 * @return tokens <ArrayList<string>>
	 * @throws IOException
	 */
	
	public static ArrayList<String> normalize(TokenStream stream) throws IOException{
		
		System.out.println("===> " + stream.getClass().getName());
		
		ArrayList<String> tokens = new ArrayList<String>();
		
		while (stream.incrementToken()) {
	         AttributeSource token = stream.cloneAttributes();
	         CharTermAttribute term =(CharTermAttribute) token.addAttribute(CharTermAttribute.class);
	         tokens.add(term.toString());
	         System.out.println(term.toString());
	     }
		
		return tokens;
		
	} // end normalize
	
	
	/**
	 * tokenize
	 * The function tokenizes a text using the StandardTokenizer, prints the tokens
	 * and returns an array list with the tokens
	 * @param String text
	 * @return ArrayList<string> tokens
	 * @throws IOException
	 */
	public static ArrayList<String> tokenize(String text) throws IOException {
		
		/*
		StandardTokenizer implements Unicode Text Segmentation 
		for breaking words apart. StandardTokenizer extends 
		Tokenizer which extends TokenStream. The API requires
		us to iterate through the tokens in the test String
		(or whatever the Reader we provide is reading from)
		by calling incrementToken().
		
		*/
		
		StringReader reader = new StringReader(text);
		StandardTokenizer tokenizer = new StandardTokenizer(Version.LUCENE_36, reader);
		
		/*
		 If you’re looking at the methods on StandardTokenizer you may have
		 realized there is no getToken() that returns a String, etc…
		 Lucene’s Filters and Tokenizers (which extend TokenStream) 
		 store attributes for each token depending on their functionality. 
		 From the StandardTokenizer we get attributes which contain the token itself,
		 the token’s type and positional information. So we need to call getAttribute
		 with the class of the Attribute we’re interested in, CharTermAttribute is
		 the one StandardTokenizer uses to contain the actual token text, etc…See the 
		 javadoc on getAttribute for additional usage details.
		 */
		
		CharTermAttribute charTermAttrib = tokenizer.getAttribute(CharTermAttribute.class);
		
		ArrayList<String> tokens = new ArrayList<String>();
		tokenizer.reset();
		
		while (tokenizer.incrementToken()) {
			tokens.add(charTermAttrib.toString());
			System.out.println(charTermAttrib.toString());
		}
		
		/*
		 In the above code, we get the attributes once and then loop through
		 tokenizer’s tokens printing out the attributes for each token.
		 Since these attributes came from the StandardTokenizer, which again 
		 is a TokenStream, you should consider that these attributes are actually
		 attributes of the stream of tokens and hence is tied to the state of tokenizer. 
		 So when we change the state of the tokenizer by calling incrementToken() the 
		 values of the attributes change as well without requiring us to manually call 
		 getAttribute again.
		 We can note that the sentence delimiters are gone, the offsets properly skip over them,
		 and the casing of the terms is preserved.
		 */
		
		tokenizer.end();
		tokenizer.close();
		
		return tokens;
		
	} // end tokenize

}
