import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class FunctionsCorenlp {
	
	/**
	 * lemmatize
	 * The function lemmatizes a string using the Stanford Tokenizer, prints the lemmas
	 * and returns an array list with the tokens
	 * @param String text
	 * @return ArrayList<string> lemmas
	 * @throws IOException
	 */
	public static ArrayList<String> lemmatize(String text) throws IOException {
		
		// list lemmas
		ArrayList<String> lemmas = new ArrayList<String>();
		
		// creates a StanfordCoreNLP object, with POS tagging, lemmatization
	    Properties props = new Properties();
	    props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
	    
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    
	    // create an empty Annotation just with the given text
	    Annotation document = new Annotation(text);
	    
	    // run all Annotators on this text
	    pipeline.annotate(document);
	    
	    // these are all the sentences in this document
	    // a CoreMap is essentially a Map that uses class objects as keys and 
	    // has values with custom types
	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	    
	    for(CoreMap sentence: sentences) {
	        // traversing the words in the current sentence
	        // a CoreLabel is a CoreMap with additional token-specific methods
	    	for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
	    	  
	    		// this is the text of the token
	    		String word = token.get(TextAnnotation.class);
	        
	    		// this is the lemma of the token
	    		String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
	    		
	    		System.out.println("token: " + "word="+word + ", lemma="+lemma);
	    		
	    		lemmas.add(lemma);
	    		
	    	} // end loop token
	    	
	    } // end loop sentence
	    
	    return lemmas;
	    
	} //end lemmatize
	
	/**
	 * tokenize
	 * The function tokenizes string using the Stanford Tokenizer, prints the lemmas
	 * and returns an array list with the tokenize
	 * @param String text
	 * @return ArrayList<string> tokenize
	 * @throws IOException
	 */
	public static ArrayList<String> tokenize(String text) throws IOException {
		
		// list tokens
		ArrayList<String> tokens = new ArrayList<String>();
		
		// creates a StanfordCoreNLP object, with POS tagging, lemmatization
	    Properties props = new Properties();
	    props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
	    
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
	    
	    // create an empty Annotation just with the given text
	    Annotation document = new Annotation(text);
	    
	    // run all Annotators on this text
	    pipeline.annotate(document);
	    
	    // these are all the sentences in this document
	    // a CoreMap is essentially a Map that uses class objects as keys and 
	    // has values with custom types
	    List<CoreMap> sentences = document.get(SentencesAnnotation.class);
	    
	    for(CoreMap sentence: sentences) {
	        // traversing the words in the current sentence
	        // a CoreLabel is a CoreMap with additional token-specific methods
	    	for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
	    	  
	    		// this is the text of the token
	    		String word = token.get(TextAnnotation.class);
	        
	    		// this is the lemma of the token
	    		String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
	    		
	    		System.out.println("token: " + "word="+word + ", lemma="+lemma);
	    		
	    		tokens.add(word);
	    		
	    	} // end loop token
	    	
	    } // end loop sentence
	    
	    return tokens;
	    
	} //end tokenize

}
