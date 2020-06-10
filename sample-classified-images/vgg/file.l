%option noyywrap
%{
#include <stdio.h>
%}
%%
auto|double|int|struct|break|else|long|switch|case|enum|register|typedef|char|extern|return|union|continue|for|signed|void|do|if|static|while|default|goto|sizeof|volatile|const|float|short {ECHO; printf(" = Keyword\n");}
[;,.'":]+ {ECHO; printf(" : Punctuation Chars\n");}

[@#$<>]+ {ECHO; printf(" : Special Chars\n");}

[\[\]{}()]+ {ECHO; printf(" : Separators\n");}

[+-/*%=&^!|]+ {ECHO; printf(" : Operators\n");}
[a-zA-Z_]+[a-zA-Z_0-9]* {ECHO; printf(" : Identifiers\n");}
[0-9]+|[0-9]+.[0-9]+ {ECHO; printf(" : Numbers\n");}
\n\n {ECHO; return 0;}

%%

int main() {
yylex();
return 0;
}
