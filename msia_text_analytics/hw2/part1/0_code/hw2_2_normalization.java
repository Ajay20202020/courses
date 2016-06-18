/** 
 * Author: Luis Steven Lin
 * 
 * Title: hw2_2_normalization
 * 
 * Purpose: Normalize using Stanford's Corenlp and Lucene's EnglishAnalyzer, PorterStemFiler, KStem
 *          
 * General Design:  Main method calls functions to read files, normalizes, print and save
 *                  a loop to get input and process inputs. 
 *
 *                  Input: wsj_0063
 *                  Output: hw2_2_normalization_EnglishAnalyzer.txt
 *                  Output: hw2_2_normalization_PorterStem.txt
 *                  Output: hw2_2_normalization_KStem.txt
 *                  Output: hw2_2_normalization_corenlp.txt
 *                  Output: hw2_2_normalization_raw.txt
 *                  Output: hw2_2_normalization_all.txt
 */

//package org.apache.lucene.analysis.standard;

import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.en.KStemFilter;
import org.apache.lucene.util.AttributeSource;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.PorterStemFilter;
import org.apache.lucene.analysis.SimpleAnalyzer;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;

import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;

import edu.stanford.nlp.util.StringUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.FileReader;

//https://code.google.com/p/poiskgoogle/source/browse/branches/parser-0.1/src/magisterarbeit/AnalysisDemo.java?spec=svn35&r=35		


public class hw2_2_normalization {
	
	
	public static void main(String[] args) throws IOException {
		
		// Inputs and Outputs
		String inputFile = "wsj_0063";
		String outputFile;
		
		// Read the file as list of strings
		ArrayList<String> lines_list = Functions.readFileLine(inputFile);
		//System.out.println(text);
		
		//test = "XY&Z Corporation – xyz@example.com";
		//final String test = "Dr. This is a test. How about that?! Huh? test@gmail.com";
		
		// declare variables
		Analyzer analyzer;
		TokenStream stream;
		
		// dictionary key = Stemmer string, value = list of normalized lines
		Map<String, ArrayList<String>> outputs = new HashMap<String, ArrayList<String>>();
		
		outputs.put("0_Raw", new ArrayList<String>());
		//outputs.put("Raw", lines_list); //has space so do it in the for loop
		
		outputs.put("1_Corenlp", new ArrayList<String>());
		outputs.put("2_EnglishAnalyzer", new ArrayList<String>());
		outputs.put("3_KStem", new ArrayList<String>());
		outputs.put("4_PorterStem", new ArrayList<String>());
		
		// dictionary key = Stemmer string, value = list of normalized lines
		Map<String, ArrayList<String>> outputs_bracket = new HashMap<String, ArrayList<String>>();
		
		outputs_bracket.put("0_Raw", new ArrayList<String>());
		//outputs.put("Raw", lines_list); //has space so do it in the for loop
		
		outputs_bracket.put("1_Corenlp", new ArrayList<String>());
		outputs_bracket.put("2_EnglishAnalyzer", new ArrayList<String>());
		outputs_bracket.put("3_KStem", new ArrayList<String>());
		outputs_bracket.put("4_PorterStem", new ArrayList<String>());
				
		
		// normalize each line using different stemmers
		for (String line:lines_list){
			
			if (!line.isEmpty()){
				
				// Raw
				outputs.get("0_Raw").add(line);
				outputs_bracket.get("0_Raw").add(line);
				
				// Corenlp
				ArrayList<String> lemmas_corenlp = FunctionsCorenlp.lemmatize(line);
				String line_lemmas  = StringUtils.join(lemmas_corenlp ," ");
				outputs.get("1_Corenlp").add(line_lemmas);
				
				ArrayList<String> lemmas_corenlp_b = Functions.addBrackets(lemmas_corenlp);
				String line_lemmas_b  = StringUtils.join(lemmas_corenlp_b ," ");
				outputs_bracket.get("1_Corenlp").add(line_lemmas_b);
				
				
				// English Analyzer tokenizes and then uses Porter Stem to normalize
				analyzer = new EnglishAnalyzer(Version.LUCENE_36);
				stream = analyzer.tokenStream("contents", new StringReader(line));
				ArrayList<String> tokens_English_Analyzer = FunctionsLucene.normalize(stream);
				String line_tokens_English_Analyzer = StringUtils.join(tokens_English_Analyzer," ");
				outputs.get("2_EnglishAnalyzer").add(line_tokens_English_Analyzer);
				
				ArrayList<String> tokens_English_Analyzer_b = Functions.addBrackets(tokens_English_Analyzer);
				String line_tokens_English_Analyzer_b = StringUtils.join(tokens_English_Analyzer_b," ");
				outputs_bracket.get("2_EnglishAnalyzer").add(line_tokens_English_Analyzer_b);
				
				// KStemFilter requires tokens first so use Simple Analyzer to tokenize
				analyzer = new SimpleAnalyzer(Version.LUCENE_36);
				stream = analyzer.tokenStream("contents", new StringReader(line));
				ArrayList<String> tokens_KStem = FunctionsLucene.normalize(new KStemFilter(stream));
				String line_tokens_KStem  = StringUtils.join(tokens_KStem ," ");
				outputs.get("3_KStem").add(line_tokens_KStem);
				
				ArrayList<String> tokens_KStem_b = Functions.addBrackets(tokens_KStem);
				String line_tokens_KStem_b  = StringUtils.join(tokens_KStem_b ," ");
				outputs_bracket.get("3_KStem").add(line_tokens_KStem_b);
				
				// PorterStemFilter requires tokens first so use Simple Analyzer to tokenize
				analyzer = new SimpleAnalyzer(Version.LUCENE_36);
				stream = analyzer.tokenStream("contents", new StringReader(line));
				ArrayList<String> tokens_PorterStem = FunctionsLucene.normalize( new PorterStemFilter(stream));
				String line_tokens_PorterStem  = StringUtils.join(tokens_PorterStem ," ");
				outputs.get("4_PorterStem").add(line_tokens_PorterStem);
				
				ArrayList<String> tokens_PorterStem_b= Functions.addBrackets(tokens_PorterStem);
				String line_tokens_PorterStem_b  = StringUtils.join(tokens_PorterStem_b ," ");
				outputs_bracket.get("4_PorterStem").add(line_tokens_PorterStem_b);
				
				
			} // end if
				
		} // end for 
		
		// Save results
		for (String key : outputs.keySet()) {
			outputFile = "hw2_2_normalization_" + key + ".txt";
			Functions.writeFile(outputFile, outputs.get(key));
		}

		// Write all together with brackets
		Functions.writeFileMap("hw2_2_normalization_all.txt", outputs_bracket);
		
		
	} // end main
	
}// end class
