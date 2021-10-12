$().ready(function () {
    // Initialize form validation on the registration form.
    // It has the name attribute "registration"
    $("#wallysearch").validate({
      // Specify validation rules
      rules: {
        // The key name on the left side is the name attribute
        // of an input field. Validation rules are defined
        // on the right side
        name: {
          required: true,
          minlength: 1
        },
      },
      messages: {
        name: "Minimum 1 character required"
      },
      
      submitHandler: function (form) {
        form.submit();
      }
    });
  });