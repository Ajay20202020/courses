/** 
 * Author: Luis Steven Lin
 * 
 * Title: hw2_1_tokenization
 * 
 * Purpose: Tokenize using Stanford's Corenlp and Lucene's Standard Tokenizer
 *          
 * General Design:  Main method calls functions to read files, normalizes, print and save
 *                  a loop to get input and process inputs. 
 *
 *                  Input: wsj_0063
 *                  Output: hw2_2_tokenization_lucene.txt
 *                  Output: hw2_2_tokenization_corenlp.txt
 *                  Output: hw2_2_tokenization_raw.txt
 *                  Output: hw2_2_tokenization_all.txt
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


public class hw2_1_tokenization {
	
	
	public static void main(String[] args) throws IOException {
		
		// Inputs and Outputs
		String inputFile = "wsj_0063";
		String outputFile;
		
		// Read the file as list of strings
		ArrayList<String> lines_list = Functions.readFileLine(inputFile);
		//System.out.println(text);
		
		//test = "XY&Z Corporation – xyz@example.com";
		//final String test = "Dr. This is a test. How about that?! Huh? test@gmail.com";
		
		// dictionary key = Method string, value = list of tokenized lines
		Map<String, ArrayList<String>> outputs = new HashMap<String, ArrayList<String>>();
		
		outputs.put("0_Raw", new ArrayList<String>());
		//outputs.put("Raw", lines_list); //has space so do it in the for loop
		
		outputs.put("1_Corenlp", new ArrayList<String>());
		outputs.put("2_Lucene", new ArrayList<String>());
		
		// Same dictionary but hold tokens in square brackets
		// dictionary key = Method string, value = list of tokenized lines
		Map<String, ArrayList<String>> outputs_bracket = new HashMap<String, ArrayList<String>>();
				
		outputs_bracket.put("0_Raw", new ArrayList<String>());
		//outputs.put("Raw", lines_list); //has space so do it in the for loop
		
		outputs_bracket.put("1_Corenlp", new ArrayList<String>());
		outputs_bracket.put("2_Lucene", new ArrayList<String>());
		
		// normalize each line using different methods
		for (String line:lines_list){
			
			if (!line.isEmpty()){
				
				// Raw
				outputs.get("0_Raw").add(line);
				outputs_bracket.get("0_Raw").add(line);
				
				// Corenlp
				ArrayList<String> tokens_corenlp = FunctionsCorenlp.tokenize(line);
				String line_tokens_corenlp  = StringUtils.join(tokens_corenlp ," ");
				outputs.get("1_Corenlp").add(line_tokens_corenlp);
				
				ArrayList<String> tokens_corenlp_bracket = Functions.addBrackets(tokens_corenlp);
				String line_tokens_corenlp_bracket  = StringUtils.join(tokens_corenlp_bracket ," ");
				outputs_bracket.get("1_Corenlp").add(line_tokens_corenlp_bracket);
				
				//Lucene
				ArrayList<String> tokens_lucene = FunctionsLucene.tokenize(line);
				String line_tokens_lucene  = StringUtils.join(tokens_lucene ," ");
				outputs.get("2_Lucene").add(line_tokens_lucene);
				
				ArrayList<String> tokens_lucene_bracket = Functions.addBrackets(tokens_lucene);
				String line_tokens_lucene_bracket  = StringUtils.join(tokens_lucene_bracket ," ");
				outputs_bracket.get("2_Lucene").add(line_tokens_lucene_bracket);
			
				
			} // end if
				
		} // end for 
		
		// Save results
		for (String key : outputs.keySet()) {
			outputFile = "hw2_1_tokenization_" + key + ".txt";
			Functions.writeFile(outputFile, outputs.get(key));
		}
		
		
		// Write all together with brackets
		Functions.writeFileMap("hw2_1_tokenization_all.txt", outputs_bracket);
		
		
		
		
	} // end main
	
}// end class
