/**
 * CommonModel class
 * It implements some useful method that can check the validity, of some data that 
 * needs to be inserted into the database
 */
export class CommonMiddleware {  
  /**
   * Function that checks if the given value is a string
   * @param value 
   * 
   * @returns true if the value is a valid string
   * @returns false if the value is not a valid string
   */
  static validateString(value: any):boolean {
    const type:string = typeof(value);
    let isValid = false;
    if(isNaN(Number(value)) && value != null && value !== '' && type === 'string'){
      isValid = true;
    }
    return isValid;
  }

  /**
   * Function that checks if the given value is a valid number
   * @param num 
   * 
   * @returns true if it is
   * @returns false if it is not
   */
  static isNumber(num: any): boolean{
    return !isNaN(parseInt(num)) && isFinite(num);
  }

  /**
   * Function that checks if the given value is a valid array of strings
   * @param value
   * 
   * @returns true if it is
   * @returns false if it is not
   */
  static validateStringArray(value: any): boolean {
    let valid = true;
    if (Array.isArray(value)) {
      value.forEach(function(item){
        if(typeof item !== 'string'){
          valid = false;
        }
      })
    }
    return valid;
  }

  /**
   * Function that checks if the given value is a string or it is undefined
   * @param value
   * 
   * @returns true if it is
   * @returns false if it is not
   */
  static stringOrUndefined(value: any): boolean {
    let valid = true;
    if (value && typeof value !== 'string'){
      valid = false;
    }
    return valid;
  }

  /**
   * Function that checks if the given value is a valid boolean
   * @param value
   * 
   * @returns true if it is
   * @returns false if it is not
   */
  static validateBoolean(value: any): boolean{
    let valid = true;
    if (!value || typeof value !== 'boolean'){
      valid = false;
    }
    return valid;
  }
}